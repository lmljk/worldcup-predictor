"""Calibration / overconfidence backtest (#3).

Hypothesis: the ensemble is overconfident — favourites win *less* often than their
predicted probability (the "upset / variance" the blurb stuffs into a crude layer).
The disciplined fix is not an ad-hoc variance knob but measured probability
calibration: temperature scaling p ∝ p^(1/T). T>1 softens (less confident),
T<1 sharpens.

Method (look-ahead free):
  1. Walk-forward DC predictions over majors (refit as_of each match).
  2. Reliability: bin by predicted prob, compare to empirical hit-rate; report ECE.
  3. Fit T on the FIRST half (chronological) by minimising log-loss, apply to the
     SECOND half, and check RPS / log-loss vs uncalibrated on that holdout.

Run:  .venv/bin/python -m skill.backtest.calibration
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..helpers.data_loader import load_results
from ..model import dixon_coles as dc
from ..model.elo import compute_elo_history
from . import metrics
from .walkforward import MAJOR


def _temp(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs, 1e-9, None) ** (1.0 / T)
    return p / p.sum(axis=1, keepdims=True)


def collect(start="2010-01-01", end="2023-12-31", refit_days=60):
    results = load_results()
    hist, _ = compute_elo_history(results)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    test = hist[(hist.date >= t0) & (hist.date <= t1)]
    test = test[test.tournament.isin(MAJOR)].dropna(subset=["home_score", "away_score"])
    test = test.sort_values("date")
    model = model_asof = None
    P, Y, D = [], [], []
    for r in test.itertuples(index=False):
        if model is None or (r.date - model_asof).days >= refit_days:
            try:
                model = dc.fit(results, as_of=r.date); model_asof = r.date
            except ValueError:
                model = None; continue
        if r.home_team not in model.attack or r.away_team not in model.attack:
            continue
        mp = dc.match_probs(model, r.home_team, r.away_team, bool(r.neutral))
        P.append([mp["p_home"], mp["p_draw"], mp["p_away"]])
        Y.append(metrics.outcome_index(int(r.home_score), int(r.away_score)))
        D.append(r.date)
    return np.array(P), np.array(Y), D


def reliability(P, Y, bins=10):
    """ECE over the flattened per-class predictions (one-vs-rest)."""
    conf = P.flatten()
    hit = np.zeros_like(P); hit[np.arange(len(Y)), Y] = 1.0
    hit = hit.flatten()
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        pc, fc = conf[m].mean(), hit[m].mean()
        rows.append((edges[i], edges[i + 1], int(m.sum()), round(pc, 3), round(fc, 3)))
        ece += m.sum() / len(conf) * abs(pc - fc)
    return rows, ece


def fit_T(P, Y, grid=np.linspace(0.7, 1.8, 45)):
    best, bestT = 1e9, 1.0
    for T in grid:
        pc = _temp(P, T)
        ll = -np.mean(np.log(np.clip(pc[np.arange(len(Y)), Y], 1e-12, None)))
        if ll < best:
            best, bestT = ll, T
    return float(bestT)


def run():
    P, Y, D = collect()
    print(f"collected n={len(Y)} walk-forward major-match predictions\n")

    rows, ece = reliability(P, Y)
    print("Reliability (one-vs-rest, 10 bins):")
    print(f"{'bin':>12} {'n':>5} {'pred':>7} {'actual':>7}  gap")
    for lo, hi, n, pc, fc in rows:
        flag = "overconfident" if pc > fc + 0.02 else ("underconf" if fc > pc + 0.02 else "")
        print(f"  [{lo:.2f},{hi:.2f}) {n:>5} {pc:>7.3f} {fc:>7.3f}  {fc-pc:+.3f} {flag}")
    print(f"ECE = {ece:.4f}\n")

    # chronological split: fit T on first half, test on second
    n = len(Y); cut = n // 2
    Ptr, Ytr, Pte, Yte = P[:cut], Y[:cut], P[cut:], Y[cut:]
    T = fit_T(Ptr, Ytr)

    def summ(Pset, Yset):
        rps = np.mean([metrics.rps(Pset[i], Yset[i]) for i in range(len(Yset))])
        ll = -np.mean(np.log(np.clip(Pset[np.arange(len(Yset)), Yset], 1e-12, None)))
        return rps, ll

    base_rps, base_ll = summ(Pte, Yte)
    cal_rps, cal_ll = summ(_temp(Pte, T), Yte)
    print(f"Fitted T={T:.3f} on first half (n={cut}); tested on second half (n={n-cut}):")
    print(f"  {'':14}{'RPS':>9}{'logloss':>10}")
    print(f"  uncalibrated {base_rps:>9.5f}{base_ll:>10.5f}")
    print(f"  T-scaled     {cal_rps:>9.5f}{cal_ll:>10.5f}")
    print(f"  Δ            {cal_rps-base_rps:>+9.5f}{cal_ll-base_ll:>+10.5f}  "
          f"({'BETTER' if cal_rps < base_rps else 'worse'} on RPS)")
    return {"n": n, "ece": ece, "T": T,
            "holdout_rps": [base_rps, cal_rps], "holdout_ll": [base_ll, cal_ll]}


if __name__ == "__main__":
    print("=== Calibration / overconfidence backtest (majors 2010-2023) ===")
    run()
