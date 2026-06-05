"""Walk-forward ablation: do defending champions underperform? ("卫冕魔咒")

Hypothesis (user / popular narrative): the holder of a trophy underperforms in the
next edition (aging core, regression, target-on-back) — so Argentina, as the 2022
World Cup holder, should be marked down for 2026.

Look-ahead free: the defending champion of edition (tour, year) is the winner of the
*previous* edition — known before a ball is kicked. The DC model is refit `as_of`
each match date. We then (a) compare the holder's model-expected win-rate to actual,
(b) sweep a penalty on the holder's λ and measure RPS on the holder-match subset, and
(c) print each holder's actual fate.

Run:  .venv/bin/python -m skill.backtest.ablation_holder
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..helpers.data_loader import load_results
from ..model import dixon_coles as dc
from . import metrics

# defending champion = winner of the PREVIOUS edition of the same tournament.
DEFENDING = {
    ("FIFA World Cup", 1994): "Germany",  ("FIFA World Cup", 1998): "Brazil",
    ("FIFA World Cup", 2002): "France",   ("FIFA World Cup", 2006): "Brazil",
    ("FIFA World Cup", 2010): "Italy",    ("FIFA World Cup", 2014): "Spain",
    ("FIFA World Cup", 2018): "Germany",  ("FIFA World Cup", 2022): "France",
    ("UEFA Euro", 1996): "Denmark",       ("UEFA Euro", 2000): "Germany",
    ("UEFA Euro", 2004): "France",        ("UEFA Euro", 2008): "Greece",
    ("UEFA Euro", 2012): "Spain",         ("UEFA Euro", 2016): "Spain",
    ("UEFA Euro", 2021): "Portugal",      ("UEFA Euro", 2024): "Italy",
    ("Copa América", 2011): "Brazil",     ("Copa América", 2015): "Uruguay",
    ("Copa América", 2016): "Chile",      ("Copa América", 2019): "Chile",
    ("Copa América", 2021): "Brazil",     ("Copa América", 2024): "Argentina",
}


def run(penalties=(0.0, 0.05, 0.10, 0.20), verbose=True):
    results = load_results().dropna(subset=["home_score", "away_score"]).copy()
    results["year"] = pd.to_datetime(results["date"]).dt.year

    rows = []          # one per holder match: model, sides, outcome, win-prob
    per_champ = []     # descriptive fate per holder
    for (tour, year), champ in sorted(DEFENDING.items()):
        ed = results[(results.tournament == tour) & (results.year == year)]
        hm = ed[(ed.home_team == champ) | (ed.away_team == champ)].sort_values("date")
        if hm.empty:
            continue
        wins = exp = 0.0
        n = 0
        for r in hm.itertuples(index=False):
            try:
                model = dc.fit(results, as_of=r.date)
            except ValueError:
                continue
            if champ not in model.attack or (r.home_team if champ == r.away_team else r.away_team) not in model.attack:
                continue
            champ_home = (r.home_team == champ)
            opp = r.away_team if champ_home else r.home_team
            if opp not in model.attack:
                continue
            mp = dc.match_probs(model, r.home_team, r.away_team, bool(r.neutral))
            outcome = metrics.outcome_index(int(r.home_score), int(r.away_score))
            p_win = mp["p_home"] if champ_home else mp["p_away"]
            won = (outcome == 0) if champ_home else (outcome == 2)
            rows.append({"champ": champ, "champ_home": champ_home, "outcome": outcome,
                         "model": model, "home": r.home_team, "away": r.away_team,
                         "neutral": bool(r.neutral)})
            exp += p_win; wins += float(won); n += 1
        if n:
            per_champ.append((f"{tour[:4]} {year}", champ, n, round(exp / n, 3),
                              round(wins / n, 3), "group-exit?" if n <= 3 else "advanced"))

    # diagnostic
    dexp = np.mean([_pwin(r) for r in rows])
    dact = np.mean([_won(r) for r in rows])

    # ablation: penalise holder's λ
    out = {"n_holder_matches": len(rows), "n_editions": len(per_champ),
           "holder_model_winrate": round(float(dexp), 4),
           "holder_actual_winrate": round(float(dact), 4), "configs": []}
    for pen in penalties:
        scored = []
        for r in rows:
            lam_m = (1 - pen) if r["champ_home"] else 1.0
            mu_m = (1 - pen) if not r["champ_home"] else 1.0
            mp = dc.match_probs(r["model"], r["home"], r["away"], r["neutral"],
                                lam_mult=lam_m, mu_mult=mu_m)
            p = np.array([mp["p_home"], mp["p_draw"], mp["p_away"]]); p = p / p.sum()
            scored.append({"probs": p, "outcome": r["outcome"], "rps": metrics.rps(p, r["outcome"]),
                           "brier": metrics.brier(p, r["outcome"]), "log_loss": metrics.log_loss(p, r["outcome"])})
        s = metrics.summarize(scored)
        out["configs"].append({"penalty": pen, "rps": s["mean_rps"], "top_pick": s["top_pick_accuracy"]})

    if verbose:
        print("Per defending champion (model-expected vs actual win-rate over its matches):")
        print(f"  {'edition':<11}{'holder':<11}{'n':>3}{'exp':>7}{'act':>7}  fate")
        for ed, ch, n, e, a, fate in per_champ:
            flag = " ←under" if a < e - 0.05 else (" ←over" if a > e + 0.05 else "")
            print(f"  {ed:<11}{ch:<11}{n:>3}{e:>7.3f}{a:>7.3f}  {fate}{flag}")
        import json
        print("\n" + json.dumps({k: out[k] for k in
              ("n_holder_matches", "n_editions", "holder_model_winrate",
               "holder_actual_winrate", "configs")}, indent=2, ensure_ascii=False))
    return out


def _pwin(r):
    mp = dc.match_probs(r["model"], r["home"], r["away"], r["neutral"])
    return mp["p_home"] if r["champ_home"] else mp["p_away"]


def _won(r):
    return (r["outcome"] == 0) if r["champ_home"] else (r["outcome"] == 2)


if __name__ == "__main__":
    print("=== Defending-champion (卫冕魔咒) ablation — WC/Euro/Copa 1994-2024 ===")
    run()
