"""Walk-forward backtest: predict each test match using only prior data.

Two design choices keep it look-ahead free:
  * ELO is taken from `compute_elo_history`, whose home_elo/away_elo columns are
    pre-match ratings by construction.
  * The Dixon-Coles model is refit on a schedule using only matches strictly
    before the match being predicted (`fit(results, as_of=match_date)`).

Baselines to beat: pure-ELO 1X2 and a naive prior. (A raw-market baseline needs
historical odds — see The Odds API; wired in M3 if purchased.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..model import dixon_coles as dc
from ..model.elo import compute_elo_history
from . import metrics

MAJOR = {
    "FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "Confederations Cup",
}


def _elo_draw_params(hist: pd.DataFrame, before: pd.Timestamp) -> tuple[float, float]:
    """Fit a simple draw curve P(draw)=dmax*exp(-(d/scale)^2) on pre-test data."""
    tr = hist[hist["date"] < before].dropna(subset=["home_score", "away_score"])
    if len(tr) < 500:
        return 0.27, 400.0
    d = (tr["home_elo"] - tr["away_elo"]).abs().to_numpy()
    is_draw = (tr["home_score"] == tr["away_score"]).to_numpy(dtype=float)
    # coarse grid search (cheap, robust, no scipy needed here)
    best, params = 1e9, (0.27, 400.0)
    for dmax in np.linspace(0.20, 0.34, 8):
        for scale in (250, 350, 450, 600):
            pred = dmax * np.exp(-((d / scale) ** 2))
            loss = np.mean((pred - is_draw) ** 2)
            if loss < best:
                best, params = loss, (float(dmax), float(scale))
    return params


def elo_1x2(rh: float, ra: float, neutral: bool, dmax: float, scale: float) -> np.ndarray:
    from ..model.elo import expected_home
    e_h = expected_home(rh, ra, neutral)
    p_draw = dmax * np.exp(-(((rh - ra) / scale) ** 2))
    p_draw = min(p_draw, 0.6)
    p_home = (1 - p_draw) * e_h
    p_away = (1 - p_draw) * (1 - e_h)
    return np.array([p_home, p_draw, p_away])


def run(
    results: pd.DataFrame,
    start: str,
    end: str,
    refit_days: int = 60,
    majors_only: bool = True,
    xi: float = 0.0010,
    importance: float = 0.0,
    verbose: bool = True,
) -> dict:
    hist, _ = compute_elo_history(results)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    test = hist[(hist["date"] >= t0) & (hist["date"] <= t1)].copy()
    if majors_only:
        test = test[test["tournament"].isin(MAJOR)]
    test = test.dropna(subset=["home_score", "away_score"]).sort_values("date")
    if test.empty:
        return {"error": "no test matches in window"}

    dmax, scale = _elo_draw_params(hist, t0)
    model = None
    model_asof = None
    dc_rows, elo_rows = [], []
    skipped = 0

    for r in test.itertuples(index=False):
        as_of = r.date
        if model is None or (as_of - model_asof).days >= refit_days:
            try:
                model = dc.fit(results, as_of=as_of, xi=xi, importance=importance)
                model_asof = as_of
            except ValueError:
                skipped += 1
                continue

        outcome = metrics.outcome_index(int(r.home_score), int(r.away_score))
        neutral = bool(r.neutral)

        # Dixon-Coles prediction
        if r.home_team in model.attack and r.away_team in model.attack:
            mp = dc.match_probs(model, r.home_team, r.away_team, neutral)
            p_dc = np.array([mp["p_home"], mp["p_draw"], mp["p_away"]])
            dc_rows.append(_score(p_dc, outcome))
        else:
            skipped += 1

        # ELO baseline
        p_elo = elo_1x2(r.home_elo, r.away_elo, neutral, dmax, scale)
        elo_rows.append(_score(p_elo, outcome))

    out = {
        "window": [start, end],
        "majors_only": majors_only,
        "xi": xi,
        "refit_days": refit_days,
        "skipped": skipped,
        "dixon_coles": metrics.summarize(dc_rows),
        "elo_baseline": metrics.summarize(elo_rows),
    }
    if verbose:
        import json
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def _score(probs: np.ndarray, outcome: int) -> dict:
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()
    return {
        "probs": probs,
        "outcome": outcome,
        "rps": metrics.rps(probs, outcome),
        "brier": metrics.brier(probs, outcome),
        "log_loss": metrics.log_loss(probs, outcome),
    }
