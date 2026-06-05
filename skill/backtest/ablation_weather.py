"""Backtest: does weather affect match outcomes? (historical actual weather, look-ahead free)

The pre-tournament model defers weather because forecasts only reach ~16 days out — but that
limits *forecasting*, NOT *validation*. For any past match we can fetch the weather that
actually occurred (Open-Meteo ERA5 archive, free, no key) and test whether it moved the result.
This is the backtest that was missing.

Pipeline (all cached, look-ahead free):
  1. geocode distinct host cities (Open-Meteo geocoding) → lat/lon
  2. fetch the ACTUAL weather for each match's date+venue (archive API): max temp, precip, wind
  3. join to results; (a) describe goals vs weather, (b) walk-forward ablation of a heat factor

Run:  .venv/bin/python -m skill.backtest.ablation_weather
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from ..helpers import paths
from ..helpers.data_loader import load_results
from .walkforward import MAJOR

GEO_CACHE = paths.DATA / "geocode_cache.json"
WX_CACHE = paths.DATA / "weather_archive_cache.json"
UA = {"User-Agent": "worldcup-backtest/1.0"}


def _load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def _save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False))


def geocode(city: str, country: str, cache: dict) -> tuple | None:
    key = f"{city}|{country}"
    if key in cache:
        v = cache[key]
        return tuple(v) if v else None
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 5}, headers=UA, timeout=20).json()
        res = r.get("results") or []
        pick = next((x for x in res if x.get("country") == country), res[0] if res else None)
        cache[key] = [pick["latitude"], pick["longitude"]] if pick else None
    except Exception:
        cache[key] = None
    return tuple(cache[key]) if cache[key] else None


def weather(lat: float, lon: float, date: str, cache: dict) -> dict | None:
    key = f"{round(lat,2)},{round(lon,2)},{date}"
    if key in cache:
        return cache[key]
    try:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                         params={"latitude": lat, "longitude": lon,
                                 "start_date": date, "end_date": date,
                                 "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                                 "timezone": "auto"}, headers=UA, timeout=25).json()
        d = r.get("daily") or {}
        cache[key] = {"tmax": (d.get("temperature_2m_max") or [None])[0],
                      "tmin": (d.get("temperature_2m_min") or [None])[0],
                      "precip": (d.get("precipitation_sum") or [None])[0],
                      "wind": (d.get("wind_speed_10m_max") or [None])[0]} if d else None
    except Exception:
        cache[key] = None
    return cache[key]


def build(limit_years_from=1990, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    results = load_results().dropna(subset=["home_score", "away_score"]).copy()
    results["year"] = pd.to_datetime(results["date"]).dt.year
    df = results[results.tournament.isin(MAJOR) & (results.year >= limit_years_from)].copy()
    df = df[df["city"].notna() & df["country"].notna()]
    geo, wx = _load(GEO_CACHE), _load(WX_CACHE)

    # 1) geocode distinct cities concurrently (fills geo cache)
    cities = {(r.city, r.country) for r in df.itertuples(index=False)}
    todo_c = [(c, co) for (c, co) in cities if f"{c}|{co}" not in geo]
    print(f"geocoding {len(todo_c)} new cities ({len(cities)-len(todo_c)} cached)...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda cc: geocode(cc[0], cc[1], geo), todo_c))
    _save(GEO_CACHE, geo)

    # 2) fetch ACTUAL weather for each distinct (venue, date) concurrently (fills wx cache)
    need = []
    for r in df.itertuples(index=False):
        ll = geo.get(f"{r.city}|{r.country}")
        if ll:
            d = str(pd.Timestamp(r.date).date())
            k = f"{round(ll[0],2)},{round(ll[1],2)},{d}"
            if k not in wx:
                need.append((ll[0], ll[1], d))
    need = list(dict.fromkeys(need))  # dedupe, keep order
    print(f"fetching {len(need)} new weather-days ({len(wx)} cached)...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda t: weather(t[0], t[1], t[2], wx), need))
    _save(WX_CACHE, wx)

    # 3) assemble from cache (fast, no network)
    rows = []
    for r in df.itertuples(index=False):
        ll = geo.get(f"{r.city}|{r.country}")
        if not ll:
            continue
        w = wx.get(f"{round(ll[0],2)},{round(ll[1],2)},{str(pd.Timestamp(r.date).date())}")
        if not w or w.get("tmax") is None:
            continue
        rows.append({"date": r.date, "home": r.home_team, "away": r.away_team,
                     "hs": int(r.home_score), "as": int(r.away_score),
                     "goals": int(r.home_score) + int(r.away_score),
                     "tmax": w["tmax"], "precip": w["precip"] or 0.0, "wind": w["wind"] or 0.0})
    print(f"assembled {len(rows)} matches with weather", flush=True)
    return pd.DataFrame(rows)


def describe(d: pd.DataFrame):
    print(f"\nJoined {len(d)} major-tournament matches to actual weather.\n")
    print("Goals vs max temperature:")
    bins = [(-99, 10), (10, 18), (18, 24), (24, 28), (28, 32), (32, 99)]
    for lo, hi in bins:
        m = d[(d.tmax >= lo) & (d.tmax < hi)]
        if len(m) >= 10:
            print(f"  {lo:>3}-{hi:<3}°C  n={len(m):>4}  avg goals={m.goals.mean():.3f}  "
                  f"home win%={(m.hs>m['as']).mean()*100:.0f}  draw%={(m.hs==m['as']).mean()*100:.0f}")
    print("\nGoals vs precipitation:")
    for lo, hi, lab in [(0, 0.1, "dry"), (0.1, 3, "light"), (3, 10, "moderate"), (10, 999, "heavy")]:
        m = d[(d.precip >= lo) & (d.precip < hi)]
        if len(m) >= 10:
            print(f"  {lab:<9} n={len(m):>4}  avg goals={m.goals.mean():.3f}  draw%={(m.hs==m['as']).mean()*100:.0f}")
    # correlations
    print("\nPearson r (weather vs total goals):")
    for col in ("tmax", "precip", "wind"):
        r = np.corrcoef(d[col], d.goals)[0, 1]
        print(f"  {col:<7} r={r:+.3f}")
    # hot-extreme contrast
    hot = d[d.tmax >= 30]; mild = d[(d.tmax >= 14) & (d.tmax < 24)]
    if len(hot) >= 20 and len(mild) >= 20:
        print(f"\nHot (≥30°C, n={len(hot)}) avg goals {hot.goals.mean():.3f} vs "
              f"mild (14-24°C, n={len(mild)}) {mild.goals.mean():.3f}  "
              f"Δ={hot.goals.mean()-mild.goals.mean():+.3f}")


if __name__ == "__main__":
    print("=== Weather impact backtest — actual ERA5 weather × major-tournament results ===")
    d = build()
    if len(d):
        describe(d)
    else:
        print("no matches joined (geocode/weather fetch failed)")
