"""Walk-forward ablation: does a rest-day differential factor beat baseline DC?

Look-ahead free: each team's rest = days since their previous match in the full
results history strictly before the match date. The shorter-rested side's lambda
is nudged down (same shape as context.py: pen = min(cap, |diff|*per_day)).

Run:  .venv/bin/python -m skill.backtest.ablation_rest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..helpers.data_loader import load_results
from ..model import dixon_coles as dc
from ..model.elo import compute_elo_history
from . import metrics
from .walkforward import MAJOR


def _rest_days(last: dict, team: str, date) -> int:
    prev = last.get(team)
    return (date - prev).days if prev else 10


def run(start="2014-01-01", end="2023-12-31", refit_days=60,
        per_day=0.015, cap=0.06, min_diff=2, verbose=True):
    results = load_results()
    hist, _ = compute_elo_history(results)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    test = hist[(hist["date"] >= t0) & (hist["date"] <= t1)]
    test = test[test["tournament"].isin(MAJOR)]
    test = test.dropna(subset=["home_score", "away_score"]).sort_values("date")

    # last-match date per team across the FULL history, walked chronologically
    last: dict = {}
    all_played = results.dropna(subset=["home_score", "away_score"]).sort_values("date")
    # pre-roll everything strictly before t0 so rest is correct at the window start
    pre = all_played[all_played["date"] < t0]
    for r in pre.itertuples(index=False):
        last[r.home_team] = r.date
        last[r.away_team] = r.date

    model = model_asof = None
    base_rows, rest_rows = [], []
    fired = 0
    test_idx = set(zip(test["date"], test["home_team"], test["away_team"]))

    for r in all_played[all_played["date"] >= t0].itertuples(index=False):
        is_test = (r.date, r.home_team, r.away_team) in test_idx
        if is_test:
            as_of = r.date
            if model is None or (as_of - model_asof).days >= refit_days:
                try:
                    model = dc.fit(results, as_of=as_of)
                    model_asof = as_of
                except ValueError:
                    model = None
            if model and r.home_team in model.attack and r.away_team in model.attack:
                outcome = metrics.outcome_index(int(r.home_score), int(r.away_score))
                neutral = bool(r.neutral)
                # baseline
                mp = dc.match_probs(model, r.home_team, r.away_team, neutral)
                base_rows.append(_score(mp, outcome))
                # rest-adjusted
                rh = _rest_days(last, r.home_team, r.date)
                ra = _rest_days(last, r.away_team, r.date)
                diff = rh - ra
                lam_m = mu_m = 1.0
                if abs(diff) >= min_diff:
                    pen = min(cap, abs(diff) * per_day)
                    if diff > 0:      # away more rested → away (mu) down
                        mu_m = 1 - pen
                    else:             # home more rested → home (lam) down
                        lam_m = 1 - pen
                    fired += 1
                mpr = dc.match_probs(model, r.home_team, r.away_team, neutral,
                                     lam_mult=lam_m, mu_mult=mu_m)
                rest_rows.append(_score(mpr, outcome))
        # advance rest clock for EVERY played match (test or not)
        last[r.home_team] = r.date
        last[r.away_team] = r.date

    out = {
        "n": len(base_rows), "fired": fired,
        "per_day": per_day, "cap": cap, "min_diff": min_diff,
        "baseline": metrics.summarize(base_rows),
        "rest_adjusted": metrics.summarize(rest_rows),
    }
    if verbose:
        b, a = out["baseline"]["mean_rps"], out["rest_adjusted"]["mean_rps"]
        print(f"per_day={per_day} cap={cap} min_diff={min_diff}  n={len(base_rows)} fired={fired}")
        print(f"  baseline RPS={b:.5f}  rest RPS={a:.5f}  Δ={a-b:+.5f}  "
              f"({'BETTER' if a < b else 'worse'})")
    return out


def _score(mp: dict, outcome: int) -> dict:
    p = np.clip(np.array([mp["p_home"], mp["p_draw"], mp["p_away"]]), 1e-9, None)
    p /= p.sum()
    return {"probs": p, "outcome": outcome, "rps": metrics.rps(p, outcome),
            "brier": metrics.brier(p, outcome), "log_loss": metrics.log_loss(p, outcome)}


if __name__ == "__main__":
    print("=== Rest-day differential ablation (walk-forward, majors 2014-2023) ===")
    for pd_ in (0.010, 0.015, 0.025, 0.040):
        for cap_ in (0.06, 0.10):
            run(per_day=pd_, cap=cap_)
