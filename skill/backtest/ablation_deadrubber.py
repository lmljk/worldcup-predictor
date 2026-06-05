"""Walk-forward ablation: do final-group-round "dead rubber" teams underperform?

Hypothesis (#1, the 战意/motivation factor): on matchday 3 of the group stage, a
team already mathematically qualified OR eliminated has reduced motivation and
underperforms its model win-probability (rests starters / lower intensity).

Look-ahead free:
  * Groups + standings are reconstructed only from matchday 1-2 results, which are
    strictly before the matchday-3 match being predicted.
  * The Dixon-Coles model is refit `as_of` each MD3 date (only prior matches).

Scope: editions with clean 4-team / top-2-advance groups (WC 1998-2022, Euro
1996-2012). Euro 2016+ (best-third format) excluded to avoid mislabeling status.

Run:  .venv/bin/python -m skill.backtest.ablation_deadrubber
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..helpers.data_loader import load_results
from ..model import dixon_coles as dc
from . import metrics

# (tournament, year) editions with 4-team groups, top-2 advance (no best-thirds).
EDITIONS = (
    [("FIFA World Cup", y) for y in (1998, 2002, 2006, 2010, 2014, 2018, 2022)]
    + [("UEFA Euro", y) for y in (1996, 2000, 2004, 2008, 2012)]
)


def _reconstruct_groups(edm: pd.DataFrame):
    """Union-find on opponents within an edition's first 3 matches per team →
    4-team round-robin groups. Returns list of (teams[4], matches_df) for clean
    groups only."""
    edm = edm.sort_values("date")
    first3: dict[str, list] = {}
    for r in edm.itertuples(index=False):
        for t, o in ((r.home_team, r.away_team), (r.away_team, r.home_team)):
            first3.setdefault(t, [])
            if len(first3[t]) < 3 and o not in first3[t]:
                first3[t].append(o)
    parent = {t: t for t in first3}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)
    for t, opps in first3.items():
        for o in opps:
            if o in parent:
                union(t, o)
    comps: dict[str, list] = {}
    for t in parent:
        comps.setdefault(find(t), []).append(t)

    groups = []
    for teams in comps.values():
        if len(teams) != 4:
            continue
        ts = set(teams)
        gm = edm[(edm.home_team.isin(ts)) & (edm.away_team.isin(ts))]
        # a clean round-robin = each of the 4 teams plays the other 3 once (6 matches)
        gm = gm.drop_duplicates(subset=["home_team", "away_team"])
        if len(gm) == 6:
            groups.append((teams, gm.sort_values("date")))
    return groups


def _standings(matches: pd.DataFrame, teams: list[str]):
    """{team: (pts, gd, gf)} from a set of played matches."""
    tab = {t: [0, 0, 0] for t in teams}
    for r in matches.itertuples(index=False):
        hs, as_ = int(r.home_score), int(r.away_score)
        tab[r.home_team][1] += hs - as_; tab[r.home_team][2] += hs
        tab[r.away_team][1] += as_ - hs; tab[r.away_team][2] += as_
        if hs > as_:
            tab[r.home_team][0] += 3
        elif hs < as_:
            tab[r.away_team][0] += 3
        else:
            tab[r.home_team][0] += 1; tab[r.away_team][0] += 1
    return {t: tuple(v) for t, v in tab.items()}


def _top2(tab: dict) -> set:
    order = sorted(tab, key=lambda t: (tab[t][0], tab[t][1], tab[t][2]))
    return set(order[-2:])


def _status(teams, played2, md3, team):
    """'clinched' / 'eliminated' / 'live' for `team` going into MD3, by enumerating
    all 3^k outcomes of the remaining MD3 matches (k=2 in a 4-team group)."""
    rem = list(md3.itertuples(index=False))
    base = _standings(played2, teams)
    top2_count = 0
    combos = list(itertools.product([0, 1, 2], repeat=len(rem)))  # 0 home, 1 draw, 2 away
    for combo in combos:
        tab = {t: list(base[t]) for t in teams}
        for res, r in zip(combo, rem):
            if res == 0:
                tab[r.home_team][0] += 3
            elif res == 2:
                tab[r.away_team][0] += 3
            else:
                tab[r.home_team][0] += 1; tab[r.away_team][0] += 1
            # crude GD/GF: assume a 1-0 / 0-0 to break ties deterministically
            if res == 0:
                tab[r.home_team][1] += 1; tab[r.home_team][2] += 1
            elif res == 2:
                tab[r.away_team][1] += 1; tab[r.away_team][2] += 1
        tabt = {t: tuple(tab[t]) for t in teams}
        if team in _top2(tabt):
            top2_count += 1
    if top2_count == len(combos):
        return "clinched"
    if top2_count == 0:
        return "eliminated"
    return "live"


def run(results=None, penalties=(0.0, 0.08, 0.15, 0.25), verbose=True):
    if results is None:
        results = load_results()
    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results["year"] = pd.to_datetime(results["date"]).dt.year

    md3_rows = []   # one per MD3 match: dict with probs, outcome, side statuses
    for tour, year in EDITIONS:
        edm = results[(results.tournament == tour) & (results.year == year)]
        if edm.empty:
            continue
        groups = _reconstruct_groups(edm)
        for teams, gm in groups:
            md3 = gm.iloc[4:6]           # last 2 of 6 matches = matchday 3
            played2 = gm.iloc[0:4]       # matchdays 1-2
            for r in md3.itertuples(index=False):
                hs = _status(teams, played2, md3, r.home_team)
                as_ = _status(teams, played2, md3, r.away_team)
                md3_rows.append({"date": r.date, "home": r.home_team, "away": r.away_team,
                                 "neutral": bool(r.neutral), "home_status": hs,
                                 "away_status": as_,
                                 "outcome": metrics.outcome_index(int(r.home_score), int(r.away_score))})

    # walk-forward DC predictions for each MD3 match
    enriched = []
    for row in md3_rows:
        try:
            model = dc.fit(results, as_of=pd.Timestamp(row["date"]))
        except ValueError:
            continue
        if row["home"] not in model.attack or row["away"] not in model.attack:
            continue
        row["model"] = model
        enriched.append(row)

    # diagnostic: do dead-rubber sides win less than the model expects?
    dead_exp, dead_act, n_dead = 0.0, 0.0, 0
    for row in enriched:
        mp = dc.match_probs(row["model"], row["home"], row["away"], row["neutral"])
        for side, status, p_win, won in (
            ("home", row["home_status"], mp["p_home"], row["outcome"] == 0),
            ("away", row["away_status"], mp["p_away"], row["outcome"] == 2)):
            if status in ("clinched", "eliminated"):
                dead_exp += p_win; dead_act += float(won); n_dead += 1

    # ablation: penalise the dead side's lambda, measure RPS on MD3 subset
    out = {"n_md3": len(enriched), "n_dead_sides": n_dead,
           "dead_model_winrate": round(dead_exp / max(n_dead, 1), 4),
           "dead_actual_winrate": round(dead_act / max(n_dead, 1), 4),
           "configs": []}
    for pen in penalties:
        rows = []
        for row in enriched:
            lam_m = mu_m = 1.0
            if row["home_status"] in ("clinched", "eliminated"):
                lam_m = 1 - pen
            if row["away_status"] in ("clinched", "eliminated"):
                mu_m = 1 - pen
            mp = dc.match_probs(row["model"], row["home"], row["away"], row["neutral"],
                                lam_mult=lam_m, mu_mult=mu_m)
            p = np.array([mp["p_home"], mp["p_draw"], mp["p_away"]])
            rows.append({"probs": p / p.sum(), "outcome": row["outcome"],
                         "rps": metrics.rps(p / p.sum(), row["outcome"]),
                         "brier": metrics.brier(p / p.sum(), row["outcome"]),
                         "log_loss": metrics.log_loss(p / p.sum(), row["outcome"])})
        s = metrics.summarize(rows)
        out["configs"].append({"penalty": pen, "rps": s["mean_rps"],
                               "top_pick": s["top_pick_accuracy"]})

    if verbose:
        import json
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    print("=== Dead-rubber (战意) ablation — MD3 of clean 4-team groups ===")
    run()
