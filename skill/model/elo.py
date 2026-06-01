"""World-football ELO, computed incrementally from match results.

Chronological pass → ratings are always as-of the match date, so using them in a
backtest introduces no look-ahead. Mirrors the eloratings.net methodology
(home advantage + margin-of-victory multiplier + tournament weight).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

# Tournament importance weight (K multiplier base), eloratings.net-style.
TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "Copa América": 50,
    "African Cup of Nations": 40,
    "AFC Asian Cup": 40,
    "Confederations Cup": 40,
    "UEFA Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 30
HOME_ADV = 65.0  # rating points added to the home side (neutral venues get 0)
BASE_RATING = 1500.0


def _k(tournament: str) -> float:
    return TOURNAMENT_K.get(tournament, DEFAULT_K)


def _mov_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def compute_elo_history(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay all matches; return per-match pre-game ratings + final rating table.

    The returned DataFrame has home_elo/away_elo = ratings BEFORE that match —
    exactly what a forecaster would have known. Final dict = latest rating per team.
    """
    ratings: dict[str, float] = defaultdict(lambda: BASE_RATING)
    rows = []
    for r in results.itertuples(index=False):
        h, a = r.home_team, r.away_team
        rh, ra = ratings[h], ratings[a]
        adv = 0.0 if getattr(r, "neutral", False) else HOME_ADV
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
        rows.append((rh, ra, exp_h))

        gd = int(r.home_score) - int(r.away_score)
        score_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k = _k(r.tournament) * _mov_multiplier(gd)
        delta = k * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    hist = results.copy()
    hist["home_elo"] = [x[0] for x in rows]
    hist["away_elo"] = [x[1] for x in rows]
    hist["exp_home"] = [x[2] for x in rows]
    return hist, dict(ratings)


def expected_home(rh: float, ra: float, neutral: bool = True) -> float:
    adv = 0.0 if neutral else HOME_ADV
    return 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
