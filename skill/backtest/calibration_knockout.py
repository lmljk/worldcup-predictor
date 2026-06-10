"""Knockout-advancement calibration: does the model over-rate big favourites?

The title-concentration question (model top1 28% vs market 16%): if knockout matches are
*per-match* over-confident for favourites, that compounds over 7 rounds into too-peaked title
odds. Test it directly on historical knockout games.

Look-ahead free: a knockout match is one where BOTH teams have already played ≥3 matches in that
edition (group stage done). Model is refit `as_of` the match date. The model's P(team advances)
mirrors the Monte Carlo's `_play`: P(scores more in regulation) + P(draw)·λ/(λ+μ) — i.e. draws
resolved by the same strength-weighted coin. Actual winner = decisive regulation result, else the
penalty-shootout winner (martj42 shootouts.csv).

Run:  .venv/bin/python -m skill.backtest.calibration_knockout
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..helpers import paths
from ..helpers.data_loader import load_results
from ..model import dixon_coles as dc
from .ablation_holder import DEFENDING  # reuse the WC/Euro/Copa edition list? no — build here

MAJOR_EDITIONS = (
    [("FIFA World Cup", y) for y in (1998, 2002, 2006, 2010, 2014, 2018, 2022)]
    + [("UEFA Euro", y) for y in (1996, 2000, 2004, 2008, 2012, 2016, 2021, 2024)]
    + [("Copa América", y) for y in (2011, 2015, 2016, 2019, 2021, 2024)]
)


def _adv_prob(model, a, b):
    """P(a advances over b) on a neutral KO: regulation win + draw resolved by strength coin."""
    mp = dc.match_probs(model, a, b, neutral=True)
    lam, mu = model.lambdas(a, b, neutral=True)
    coin = lam / (lam + mu)
    return mp["p_home"] + mp["p_draw"] * coin


def run(verbose=True):
    results = load_results().dropna(subset=["home_score", "away_score"]).copy()
    results["year"] = pd.to_datetime(results["date"]).dt.year
    sh = pd.read_csv(paths.HISTORICAL / "shootouts.csv")
    sh["date"] = pd.to_datetime(sh["date"]).dt.date
    shoot = {(r.date, frozenset((r.home_team, r.away_team))): r.winner for r in sh.itertuples()}

    rows = []
    for tour, year in MAJOR_EDITIONS:
        ed = results[(results.tournament == tour) & (results.year == year)].sort_values("date")
        played = {}
        for r in ed.itertuples(index=False):
            ha, aa = played.get(r.home_team, 0), played.get(r.away_team, 0)
            if ha >= 3 and aa >= 3:          # both done with group stage → knockout
                rows.append(r)
            played[r.home_team] = ha + 1
            played[r.away_team] = aa + 1

    model = model_asof = None
    P, Y = [], []   # P = favourite's predicted advance prob; Y = favourite actually advanced
    for r in sorted(rows, key=lambda r: r.date):
        d = pd.Timestamp(r.date)
        if model is None or (d - model_asof).days >= 60:
            try:
                model = dc.fit(results, as_of=d); model_asof = d
            except ValueError:
                model = None; continue
        if r.home_team not in model.attack or r.away_team not in model.attack:
            continue
        pa = _adv_prob(model, r.home_team, r.away_team)
        fav, fav_p = (r.home_team, pa) if pa >= 0.5 else (r.away_team, 1 - pa)
        hs, as_ = int(r.home_score), int(r.away_score)
        if hs != as_:
            winner = r.home_team if hs > as_ else r.away_team
        else:
            winner = shoot.get((pd.Timestamp(r.date).date(),
                                frozenset((r.home_team, r.away_team))))
            if winner is None:
                continue   # drawn KO with no shootout record → skip
        P.append(fav_p); Y.append(1.0 if winner == fav else 0.0)

    P, Y = np.array(P), np.array(Y)
    if verbose:
        print(f"n knockout matches = {len(P)}\n")
        print("Favourite-advances calibration (binned by model's predicted advance prob):")
        print(f"  {'bin':>12} {'n':>4} {'pred':>7} {'actual':>7}  gap")
        edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        ece = 0.0
        for i in range(len(edges) - 1):
            m = (P >= edges[i]) & (P < edges[i + 1])
            if m.sum() == 0:
                continue
            pc, fc = P[m].mean(), Y[m].mean()
            ece += m.sum() / len(P) * abs(pc - fc)
            flag = "OVER-confident" if pc > fc + 0.03 else ("under" if fc > pc + 0.03 else "ok")
            print(f"  [{edges[i]:.1f},{edges[i+1]:.2f}) {m.sum():>4} {pc:>7.3f} {fc:>7.3f}  {fc-pc:+.3f} {flag}")
        print(f"\noverall: predicted favourites win {P.mean():.3f}, actually {Y.mean():.3f} "
              f"(Δ {Y.mean()-P.mean():+.3f}); ECE {ece:.4f}")
    return {"n": len(P), "pred": float(P.mean()), "actual": float(Y.mean())}


if __name__ == "__main__":
    print("=== Knockout favourite-advancement calibration (WC/Euro/Copa) ===")
    run()
