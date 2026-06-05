"""Pre-tournament injury prior (transparent, NOT a backtested weight).

A player confirmed OUT for the whole tournament is removed from his team's squad
*before* squad-talent / FC25 strength is computed. The projected XI then rebuilds
with the next-best player, so the team's attack/defence fall by exactly the
absentee's **marginal** contribution to the XI. The magnitude is therefore
endogenous (a recomputed lineup), not a hand-tuned penalty — consistent with the
no-overfit discipline. It is surfaced on the dashboard as a factor and labelled a
prior, never a fitted parameter, because (like the talent/context priors) it can't
be walk-forward validated on free data (no historical injury+counterfactual series).

Distinct from the match-day lineup-absence channel (`players.match_scorers`), which
adjusts a *single match's* goal shares; this adjusts season-long team strength and
the Monte Carlo across all of that team's matches.
"""
from __future__ import annotations

import re
import unicodedata


def _nm(s: str) -> str:
    return re.sub(r"[^a-z]", "",
                  unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower())


def apply(squads: dict, injuries: list) -> tuple[dict, dict]:
    """Drop tournament-long absentees from each squad.

    Returns (filtered_squads, removed) where removed = {team: [records...]} for the
    players actually matched and removed (so callers can show + measure the effect).
    """
    by_team: dict[str, list] = {}
    for inj in injuries or []:
        if not isinstance(inj, dict) or not inj.get("player") or not inj.get("team"):
            continue
        if inj.get("status", "out_tournament") != "out_tournament":
            continue
        by_team.setdefault(inj["team"], []).append(inj)

    out, removed = {}, {}
    for tm, ps in squads.items():
        injs = by_team.get(tm)
        if not injs:
            out[tm] = ps
            continue
        keys = {_nm(i["player"]): i for i in injs}
        kept, gone = [], []
        for p in ps:
            hit = keys.get(_nm(p.get("name", "")))
            if hit:
                gone.append({**hit, "matched": p.get("name")})
            else:
                kept.append(p)
        out[tm] = kept
        if gone:
            removed[tm] = gone
    return out, removed


def exclude_keys(removed: dict) -> dict:
    """{team: {normalised names}} so FC25's all-nationals fallback also drops them."""
    return {tm: {_nm(g.get("matched") or g["player"]) for g in gs} for tm, gs in removed.items()}
