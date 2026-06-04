"""EA Sports FC 25 player ratings → multi-dimensional national-team strength.

(FC 26 isn't released until ~Sept 2026, after the World Cup, so FC 25 is the latest
available game.) For each national team we take its WC squad, match players to their
FC 25 ratings, pick a projected best XI (4-3-3 by OVR within position groups), and compute
attack / defence / overall ratings — forwards weighted on attacking sub-attributes,
defenders/keeper on defensive ones, so a team's attack-vs-defence balance is captured.

Like the clubelo talent prior, this is a current-snapshot quality proxy — it can't be
walk-forward validated, so it's blended modestly with clubelo, not treated as a fitted factor.
"""
from __future__ import annotations

import re
import statistics
import unicodedata

import pandas as pd

from ..helpers import paths

FC25_CSV = paths.DATA / "fc25_players.csv"

# FC25 Nation label -> our dataset team name (where they differ).
NATION_ALIAS = {
    "Korea Republic": "South Korea", "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran",
    "Czechia": "Czech Republic", "Türkiye": "Turkey", "USA": "United States",
}
POS_GROUP = {  # FC25 Position -> group
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MF", "CM": "MF", "CAM": "MF", "LM": "MF", "RM": "MF",
    "ST": "FW", "CF": "FW", "LW": "FW", "RW": "FW",
}
FORMATION = {"GK": 1, "DEF": 4, "MF": 3, "FW": 3}  # projected 4-3-3
# how much each group contributes to team attack / defence
ATK_W = {"FW": 1.0, "MF": 0.6, "DEF": 0.2, "GK": 0.0}
DEF_W = {"GK": 1.0, "DEF": 1.0, "MF": 0.5, "FW": 0.2}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


FC25_SOURCE = ("https://raw.githubusercontent.com/Rayen-khammessi/"
               "Football-Data-Engineering-Analytics-Project-EA-Sports-FC-25/HEAD/male_players.csv")


def load_fc25() -> pd.DataFrame:
    if not FC25_CSV.exists():  # self-fetch (regenerable; dataset is gitignored)
        import requests
        FC25_CSV.write_bytes(requests.get(FC25_SOURCE, timeout=60).content)
    df = pd.read_csv(FC25_CSV)
    df["nation_std"] = df["Nation"].map(lambda x: NATION_ALIAS.get(x, x))
    df["grp"] = df["Position"].map(POS_GROUP).fillna("MF")
    df["nkey"] = df["Name"].map(_norm)
    return df


def _atk(p) -> float:
    return float((p.get("SHO", 0) + p.get("DRI", 0) + p.get("PAS", 0) + p.get("PAC", 0)) / 4)


def _def(p) -> float:
    if p["grp"] == "GK":
        gk = [p.get(k, 0) for k in ("GK Diving", "GK Positioning", "GK Reflexes", "GK Handling")]
        return float(sum(gk) / 4) if any(gk) else float(p.get("OVR", 0))
    return float((p.get("DEF", 0) + p.get("PHY", 0)) / 2)


def team_ratings(squads: dict) -> dict[str, dict]:
    """Per-team FC25 attack/defence/overall from a projected best XI, plus per-player OVR."""
    fc = load_fc25()
    out, player_ovr = {}, {}
    for team, sq in squads.items():
        sub = fc[fc["nation_std"] == team]
        if sub.empty:
            continue
        # match squad players to FC25 by normalised name; fall back to all FC25 nationals
        names = {_norm(p["name"]) for p in sq}
        m = sub[sub["nkey"].isin(names)]
        pool = m if len(m) >= 11 else sub
        for r in pool.itertuples(index=False):
            player_ovr[(team, r.Name)] = {"ovr": int(getattr(r, "OVR", 0)),
                                          "pos": r.grp, "league": getattr(r, "League", "")}
        # projected XI: best OVR within each position group per the formation
        xi = []
        for grp, k in FORMATION.items():
            g = pool[pool["grp"] == grp].sort_values("OVR", ascending=False).head(k)
            xi.extend(g.to_dict("records"))
        if len(xi) < 7:
            continue
        atk = _wmean([_atk(p) for p in xi], [ATK_W[p["grp"]] for p in xi])
        dfn = _wmean([_def(p) for p in xi], [DEF_W[p["grp"]] for p in xi])
        out[team] = {
            "fc_overall": round(statistics.mean([p["OVR"] for p in xi]), 1),
            "fc_attack": round(atk, 1), "fc_defence": round(dfn, 1),
            "xi_size": len(xi),
        }
    # z-scores across teams
    for key in ("fc_overall", "fc_attack", "fc_defence"):
        vals = [v[key] for v in out.values()]
        mu, sd = statistics.mean(vals), statistics.pstdev(vals) or 1.0
        for v in out.values():
            v[key + "_z"] = round((v[key] - mu) / sd, 3)
    return out, player_ovr


def _wmean(vals, ws):
    s = sum(ws) or 1.0
    return sum(v * w for v, w in zip(vals, ws)) / s
