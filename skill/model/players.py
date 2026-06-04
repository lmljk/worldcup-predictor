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


SCORER_RATE = 0.18  # career goals/cap above which a player is a "regular scorer"


def _weight(p: dict) -> float:
    """Goal-share weight = career rate × form factor × penalty-taker boost.

    Form factor (from goalscorers recent-2y goals):
      * hot scorers get boosted;
      * a player who WAS a scorer (career rate ≥ SCORER_RATE) but has gone cold
        (<3 recent goals) is pulled DOWN — backtest: once-strong scorers who went cold
        score ~45% fewer future goals than those who stayed hot (3.46 vs 6.30).
    This handles fading/peripheral attackers (e.g. Neymar) without an age factor; the
    exact "is he even playing" signal is applied separately from match-day lineups.
    """
    base = player_rate(p)
    rg = p.get("recent_goals", 0)
    if base >= SCORER_RATE and rg < 3:
        form_mult = 0.55 + 0.15 * rg            # cold: 0.55 / 0.70 / 0.85
    else:
        form_mult = min(1.5, 1.0 + 0.06 * rg)   # hot boost
    pen_mult = 1.15 if p.get("pen_taker") else 1.0
    ovr = p.get("fc_ovr")  # EA FC25 ability (validated to add signal for goals)
    ovr_mult = min(1.30, max(0.80, 1.0 + 0.10 * ((ovr - 78) / 8))) if ovr else 1.0
    league_mult = _league_mult(p.get("league", ""))  # league level
    return base * form_mult * pen_mult * ovr_mult * league_mult


# League strength tiers (FC25 League names, substring match) for the awards layer.
TOP5_LEAGUES = ("premier league", "laliga", "la liga", "bundesliga", "serie a", "ligue 1")
TIER2_LEAGUES = ("eredivisie", "liga portugal", "primeira", "saudi", "mls", "championship",
                 "süper lig", "super lig", "belgian", "jupiler", "scottish")


def _league_mult(league: str) -> float:
    l = (league or "").lower()
    if any(k in l for k in TOP5_LEAGUES):
        return 1.06
    if any(k in l for k in TIER2_LEAGUES):
        return 1.0
    return 0.95 if l else 1.0


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


def match_scorers(lam_team: float, squad: list[dict], topn: int = 5,
                  absent: set | None = None) -> list[dict]:
    """Top likely scorers for one side given that side's expected goals.

    `absent` = players confirmed NOT in today's XI (from match-day lineups) — they are
    dropped and their share redistributed, the clean 'is he actually playing' signal that
    recent-form alone can't give. Empty pre-match (lineups publish ~1h before kickoff)."""
    import math

    absent = absent or set()
    shares = [(n, pos, s) for n, pos, s in goal_shares(squad) if n not in absent]
    tot = sum(s for _, _, s in shares) or 1.0
    shares = [(n, pos, s / tot) for n, pos, s in shares]
    out = []
    for name, pos, share in shares:
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
