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


def goal_shares(squad: list[dict]) -> list[tuple[str, str, float]]:
    """Return [(name, pos, share)] with shares summing to 1 over the squad."""
    rates = [(p["name"], p.get("pos", "MF"), player_rate(p)) for p in squad]
    tot = sum(r for _, _, r in rates) or 1.0
    return [(n, pos, r / tot) for n, pos, r in rates]


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
