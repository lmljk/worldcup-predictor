"""Deterministic 'most-likely' knockout projection — the modal bracket (出线树).

A single-path point projection used for the dashboard's elimination tree:

  1. each group is resolved to a winner / runner-up / third by the model's
     expected round-robin points (3·P(win) + 1·P(draw), neutral venue);
  2. the 8 best third-placed teams (by expected points) are slotted into the
     official 2026 R32 bracket via the same eligibility map the Monte Carlo uses;
  3. every knockout tie is won by the side with the higher model win-probability
     (draws split 50/50, since knockouts can't end level), climbing R32 → champion.

This is a *point projection*, not the Monte Carlo title odds. The champion here is
the single most-likely path; `title_probability` (which integrates over every
possible path and upset) stays the headline number — the two can legitimately
differ, and the dashboard shows both.
"""
from __future__ import annotations

from itertools import combinations

from ..model import dixon_coles as dc
from .montecarlo import (
    OFFICIAL_GROUPS,
    _GI,
    _R32,
    _THIRD_ELIG,  # noqa: F401  (kept for parity / future eligibility checks)
    _THIRD_POS,
    _assign_thirds,
)

_ROUND_NAMES = ["R32", "R16", "QF", "SF", "Final"]


def _eff_win(model, a: str, b: str) -> float:
    """P(a beats b) on the day at a neutral venue; a draw is split 50/50."""
    mp = dc.match_probs(model, a, b, neutral=True)
    return mp["p_home"] + 0.5 * mp["p_draw"]


def _group_table(model, teams: list[str]):
    """Expected league points for each team in a round-robin. Returns the teams
    ordered best-first plus the {team: xpts} map."""
    xpts = {t: 0.0 for t in teams}
    for a, b in combinations(teams, 2):
        mp = dc.match_probs(model, a, b, neutral=True)
        xpts[a] += 3 * mp["p_home"] + mp["p_draw"]
        xpts[b] += 3 * mp["p_away"] + mp["p_draw"]
    order = sorted(teams, key=lambda t: -xpts[t])
    return order, xpts


def project(model, fixtures=None) -> dict:
    """Build the single most-likely bracket from the fitted model.

    `fixtures` is accepted for signature parity with montecarlo.run but the
    official A..L draw is authoritative here, so it is unused.
    """
    winners, runners, thirds = {}, {}, {}
    for code, teams in OFFICIAL_GROUPS.items():
        order, xpts = _group_table(model, teams)
        winners[code] = order[0]
        runners[code] = order[1]
        thirds[code] = (order[2], xpts[order[2]])

    # 8 best third-placed groups by expected points → official slot assignment
    best_thirds = sorted(thirds, key=lambda c: -thirds[c][1])[:8]
    qual = tuple(sorted(_GI[c] for c in best_thirds))
    assign = _assign_thirds(qual)            # group-index per third-slot (len 8)
    gi_team = {_GI[c]: thirds[c][0] for c in OFFICIAL_GROUPS}

    # fill the 32 bracket positions in official R32 order
    slots: list[str | None] = [None] * 32
    for mi, (h, a) in enumerate(_R32):
        for si, slot in enumerate((h, a)):
            pos = mi * 2 + si
            if slot[0] == "W":
                slots[pos] = winners[slot[1]]
            elif slot[0] == "RU":
                slots[pos] = runners[slot[1]]
            # '3RD' positions filled next
    for k, pos in enumerate(_THIRD_POS):
        slots[pos] = gi_team[assign[k]]

    # climb the single-elimination tree, modal winner each tie
    cur: list[str] = list(slots)             # type: ignore[arg-type]
    rounds = []
    for rname in _ROUND_NAMES:
        matches, nxt = [], []
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i + 1]
            pa = _eff_win(model, a, b)
            win = a if pa >= 0.5 else b
            matches.append({
                "a": a, "b": b, "winner": win,
                "p": round(pa if win == a else 1 - pa, 4),
            })
            nxt.append(win)
        rounds.append({"round": rname, "matches": matches})
        cur = nxt

    return {
        "champion": cur[0],
        "rounds": rounds,
        "group_winners": winners,
        "group_runners": runners,
    }
