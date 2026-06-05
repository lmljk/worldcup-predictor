"""Backtest: do cold-climate teams underperform in extreme heat? (climate mismatch)

The steelman for weather after the aggregate null (Run 19): weather may not move goals on
average, but a *differential* could exist — a cold-climate side stressed by heat its
opponent is acclimatised to (the heat analogue of altitude).

Design (look-ahead free):
  * each national team gets a home-climate baseline = mean daily max temp of its country
    (one ERA5 climatology year, stable climatology, no leakage);
  * for each major-tournament match (actual match-day temp from Run 19's cache) compute each
    side's heat mismatch = match_tmax − team_baseline;
  * walk-forward: refit DC as_of; test whether the MORE heat-mismatched side wins less than
    the model predicts, and whether penalising its λ by the mismatch gap improves RPS.

Run:  .venv/bin/python -m skill.backtest.ablation_climate_mismatch
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from ..helpers import paths
from ..model import dixon_coles as dc
from . import metrics
from .ablation_weather import GEO_CACHE, build
from .ablation_weather import geocode as _geocode

CLIMATE_CACHE = paths.DATA / "country_climate_cache.json"
UA = {"User-Agent": "worldcup-backtest/1.0"}


def _load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def country_climate(team: str, geo: dict, clim: dict) -> float | None:
    """Mean daily-max temp for a national team's country (one climatology year)."""
    if team in clim:
        return clim[team]
    ll = _geocode(team, "", geo) or geocode_country(team, geo)
    if not ll:
        clim[team] = None
        return None
    try:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                         params={"latitude": ll[0], "longitude": ll[1],
                                 "start_date": "2019-01-01", "end_date": "2019-12-31",
                                 "daily": "temperature_2m_max", "timezone": "auto"},
                         headers=UA, timeout=30).json()
        vals = [v for v in (r.get("daily") or {}).get("temperature_2m_max", []) if v is not None]
        clim[team] = round(sum(vals) / len(vals), 2) if vals else None
    except Exception:
        clim[team] = None
    return clim[team]


def geocode_country(name: str, geo: dict):
    key = f"COUNTRY|{name}"
    if key in geo:
        v = geo[key]
        return tuple(v) if v else None
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": name, "count": 1}, headers=UA, timeout=20).json()
        res = (r.get("results") or [None])[0]
        geo[key] = [res["latitude"], res["longitude"]] if res else None
    except Exception:
        geo[key] = None
    return tuple(geo[key]) if geo[key] else None


def run(min_gap=6.0, penalties=(0.0, 0.01, 0.02, 0.04), verbose=True):
    from ..helpers.data_loader import load_results
    wx = build()  # joined match weather (cached, fast)
    teams = sorted(set(wx.home) | set(wx.away))
    geo, clim = _load(GEO_CACHE), _load(CLIMATE_CACHE)
    print(f"computing climate baselines for {len(teams)} teams...", flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda t: country_climate(t, geo, clim), teams))
    GEO_CACHE.write_text(json.dumps(geo, ensure_ascii=False))
    CLIMATE_CACHE.write_text(json.dumps(clim, ensure_ascii=False))

    base = {t: clim.get(t) for t in teams}
    sample = {t: base[t] for t in list(base)[:6]}
    print("sample baselines:", {k: v for k, v in sample.items()}, flush=True)

    results = load_results().dropna(subset=["home_score", "away_score"])
    wx = wx.sort_values("date")
    model = model_asof = None
    rows = []
    for r in wx.itertuples(index=False):
        bh, ba = base.get(r.home), base.get(r.away)
        if bh is None or ba is None:
            continue
        mm_h, mm_a = r.tmax - bh, r.tmax - ba       # heat mismatch per side
        gap = mm_h - mm_a                            # >0: home more heat-stressed than away
        if abs(gap) < min_gap:
            continue
        d = pd.Timestamp(r.date)
        if model is None or (d - model_asof).days >= 60:
            try:
                model = dc.fit(results, as_of=d); model_asof = d
            except ValueError:
                model = None; continue
        if r.home not in model.attack or r.away not in model.attack:
            continue
        outcome = metrics.outcome_index(r.hs, r.goals - r.hs)
        rows.append({"model": model, "home": r.home, "away": r.away, "outcome": outcome,
                     "gap": gap})

    # diagnostic: does the more heat-stressed side win less than the model says?
    exp = act = 0.0
    for x in rows:
        mp = dc.match_probs(x["model"], x["home"], x["away"], neutral=True)
        if x["gap"] > 0:   # home more stressed
            exp += mp["p_home"]; act += float(x["outcome"] == 0)
        else:              # away more stressed
            exp += mp["p_away"]; act += float(x["outcome"] == 2)
    n = len(rows)
    out = {"n": n, "min_gap": min_gap,
           "stressed_model_winrate": round(exp / max(n, 1), 4),
           "stressed_actual_winrate": round(act / max(n, 1), 4), "configs": []}

    # ablation: penalise the more heat-stressed side's λ ∝ the gap
    for pen in penalties:
        scored = []
        for x in rows:
            lam_m = mu_m = 1.0
            scale = 1 - min(0.20, pen * abs(x["gap"]))   # bigger gap → bigger cut
            if x["gap"] > 0:
                lam_m = scale
            else:
                mu_m = scale
            mp = dc.match_probs(x["model"], x["home"], x["away"], neutral=True,
                                lam_mult=lam_m, mu_mult=mu_m)
            p = np.array([mp["p_home"], mp["p_draw"], mp["p_away"]]); p = p / p.sum()
            scored.append({"probs": p, "outcome": x["outcome"], "rps": metrics.rps(p, x["outcome"]),
                           "brier": metrics.brier(p, x["outcome"]), "log_loss": metrics.log_loss(p, x["outcome"])})
        s = metrics.summarize(scored)
        out["configs"].append({"penalty_per_degC": pen, "rps": s["mean_rps"]})

    if verbose:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    print("=== Climate-mismatch ablation — cold-climate teams in heat ===")
    run()
