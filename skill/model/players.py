"""Player-level goal model.

Idea: the team's expected goals (lambda) for a match comes from the Dixon-Coles
model. We split that lambda across the squad by each player's *goal share* — derived
from career international scoring rate (goals/caps), shrunk toward a positional prior
so low-cap players aren't noisy. Then per-match scoring is Poisson(player_lambda),
and the Golden Boot race is a Monte Carlo accumulation across the tournament.

Career-rate priors are a reasonable v0; they sharpen once live tournament goals
arrive (the review loop tracks the real scorer table).
"""
from __future__ import annotations

# Positional prior scoring rate (goals per appearance) — shrinkage target.
POS_PRIOR = {"FW": 0.42, "MF": 0.12, "DF": 0.04, "GK": 0.0}
SHRINK = 15.0  # pseudo-appearances


def player_rate(p: dict) -> float:
    pos = p.get("pos", "MF")
    if pos == "GK":
        return 0.0
    prior = POS_PRIOR.get(pos, 0.12)
    caps = max(int(p.get("caps", 0)), 0)
    goals = max(int(p.get("goals", 0)), 0)
    return (goals + SHRINK * prior) / (caps + SHRINK)


def _weight(p: dict) -> float:
    """Goal-share weight = career rate × recent-form boost × penalty-taker boost.

    Recent form and penalty duties come from the goalscorers dataset (see enrich_form).
    Both are modest, transparent multipliers on top of the career scoring rate.
    """
    base = player_rate(p)
    form_mult = min(1.5, 1.0 + 0.06 * p.get("recent_goals", 0))   # in-form scorers
    pen_mult = 1.15 if p.get("pen_taker") else 1.0                 # penalty taker
    return base * form_mult * pen_mult


def goal_shares(squad: list[dict]) -> list[tuple[str, str, float]]:
    """Return [(name, pos, share)] with shares summing to 1 over the squad."""
    rates = [(p["name"], p.get("pos", "MF"), _weight(p)) for p in squad]
    tot = sum(r for _, _, r in rates) or 1.0
    return [(n, pos, r / tot) for n, pos, r in rates]


def enrich_form(squads: dict, goalscorers, since) -> dict:
    """Annotate each squad player with recent international goals + penalty-taker flag.

    Matched by normalised name within the player's national team. Mutates and returns squads.
    """
    import re
    import unicodedata

    def norm(s):
        s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]", "", s.lower())

    gs = goalscorers[(goalscorers["own_goal"] == False)]  # noqa: E712
    recent = gs[gs["date"] >= since]
    # per (team, normalised scorer): recent goal count
    rec_counts, pen_counts = {}, {}
    for r in recent.itertuples(index=False):
        rec_counts[(r.team, norm(r.scorer))] = rec_counts.get((r.team, norm(r.scorer)), 0) + 1
    for r in gs.itertuples(index=False):  # penalty takers from full history
        if r.penalty:
            pen_counts[(r.team, norm(r.scorer))] = pen_counts.get((r.team, norm(r.scorer)), 0) + 1
    for team, ps in squads.items():
        for p in ps:
            key = (team, norm(p["name"]))
            p["recent_goals"] = rec_counts.get(key, 0)
            p["pen_taker"] = pen_counts.get(key, 0) >= 3
    return squads


def match_scorers(lam_team: float, squad: list[dict], topn: int = 5) -> list[dict]:
    """Top likely scorers for one side given that side's expected goals."""
    import math

    out = []
    for name, pos, share in goal_shares(squad):
        lam_p = lam_team * share
        if lam_p <= 0:
            continue
        out.append({
            "name": name, "pos": pos,
            "exp_goals": round(lam_p, 3),
            "p_score": round(1 - math.exp(-lam_p), 4),
        })
    out.sort(key=lambda x: -x["exp_goals"])
    return out[:topn]
