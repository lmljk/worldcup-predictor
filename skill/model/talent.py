"""Squad-talent prior from club strength (clubelo.com).

Transfermarkt market value is the ideal "roster talent" signal, but it is bot-protected
and not reliably free-scrapable. Club Elo is an excellent free proxy: players at elite
clubs (Real Madrid, Man City, PSG, Bayern) carry high market value, so the average club
Elo of a national squad tracks its talent — capturing exactly what international-results
ELO misses (e.g. France, whose recent form lags its squad quality).

NOTE: this is a *current-snapshot prior*, not a walk-forward-validated factor (we would
need historical club-Elo snapshots to backtest it). It is applied as a small, transparent
nudge to team attack/defence and surfaced as a factor on the dashboard.
"""
from __future__ import annotations

import re
import statistics
import unicodedata

# Wikipedia club name (normalised) -> clubelo club name (normalised).
ALIAS = {
    "manchestercity": "mancity", "manchesterunited": "manunited", "bayernmunich": "bayern",
    "psveindhoven": "psv", "parissaintgermain": "parissg", "internazionale": "inter",
    "tottenhamhotspur": "tottenham", "wolverhamptonwanderers": "wolves",
    "borussiadortmund": "dortmund", "borussiamonchengladbach": "gladbach",
    "newcastleunited": "newcastle", "brightonhovealbion": "brighton",
    "westhamunited": "westham", "realbetis": "betis", "bayer04leverkusen": "leverkusen",
    "bayerleverkusen": "leverkusen", "rbleipzig": "leipzig", "acmilan": "milan",
    "asroma": "roma", "sscnapoli": "napoli", "olympiquelyonnais": "lyon",
    "olympiquemarseille": "marseille", "asmonaco": "monaco", "sevillafc": "sevilla",
    "villarrealcf": "villarreal", "athleticbilbao": "athletic", "realsociedad": "sociedad",
    "sportinglisbon": "sporting", "sportingcp": "sporting", "slbenfica": "benfica",
    "fcporto": "porto", "ajax": "ajax", "feyenoord": "feyenoord", "celticfc": "celtic",
    "rangersfc": "rangers", "galatasaray": "galatasaray", "fenerbahce": "fenerbahce",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _index(club_elo: dict[str, float]) -> dict[str, float]:
    return {_norm(k): v for k, v in club_elo.items()}


def _match(club: str, idx: dict[str, float]) -> float | None:
    n = _norm(club)
    if not n:
        return None
    if n in idx:
        return idx[n]
    if n in ALIAS and ALIAS[n] in idx:
        return idx[ALIAS[n]]
    for k, v in idx.items():
        if len(k) > 4 and (k in n or n in k):
            return v
    return None


def squad_talent(squads: dict, club_elo: dict, min_matched: int = 4,
                 default: float = 1400.0) -> dict[str, dict]:
    """Per-team talent = mean club Elo of squad players matched to clubelo."""
    idx = _index(club_elo)
    raw, matched = {}, {}
    for tm, ps in squads.items():
        vals = [m for p in ps if (m := _match(p.get("club", ""), idx)) is not None]
        matched[tm] = len(vals)
        raw[tm] = (sum(vals) / len(vals)) if len(vals) >= min_matched else default
    mean = statistics.mean(raw.values())
    sd = statistics.pstdev(raw.values()) or 1.0
    return {tm: {"talent": round(v, 0), "talent_z": round((v - mean) / sd, 3),
                 "matched": matched[tm]} for tm, v in raw.items()}


def adjusted_strength(attack: dict, defence: dict, talent: dict, weight: float = 0.10):
    """Nudge attack & defence by talent z-score (talented squads attack better, concede less)."""
    atk = dict(attack)
    dfc = dict(defence)
    for tm, t in talent.items():
        z = t.get("talent_z", 0.0)
        if tm in atk:
            atk[tm] += weight * z
            dfc[tm] += weight * z
    return atk, dfc


def combined_adjust(attack: dict, defence: dict, clubelo_talent: dict,
                    fc_ratings: dict, weight: float = 0.10):
    """Blend two quality proxies (clubelo club-Elo + EA FC25 squad rating) into the
    attack/defence nudge. FC25's attack/defence split lets us nudge each side separately:
    attacking-strong squads boost their goal rate, defensively-strong squads concede less."""
    atk, dfc = dict(attack), dict(defence)
    for tm in atk:
        cz = clubelo_talent.get(tm, {}).get("talent_z", 0.0)
        fc = fc_ratings.get(tm, {})
        fo = fc.get("fc_overall_z")
        overall_z = (cz + fo) / 2 if fo is not None else cz  # consensus of the two proxies
        fa = fc.get("fc_attack_z", overall_z)
        fd = fc.get("fc_defence_z", overall_z)
        atk[tm] += weight * (0.5 * overall_z + 0.5 * fa)
        dfc[tm] += weight * (0.5 * overall_z + 0.5 * fd)
    return atk, dfc
