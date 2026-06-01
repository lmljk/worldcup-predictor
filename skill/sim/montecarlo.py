"""Vectorised Monte Carlo tournament simulation for WC2026.

Group stage uses the real fixture list (reconstructed group structure). Knockout
is a single-elimination bracket among the 32 advancers. Everything is vectorised
across `n` simulations with numpy, so 50k runs take a couple of seconds.

Scorelines are sampled from the Dixon-Coles expected goals (independent Poisson;
the DC low-score correction is a likelihood term, negligible for sampling). Knockout
draws are resolved by a strength-weighted penalty coin flip.

Note: the knockout bracket is randomised per simulation rather than using FIFA's
exact slot mapping for the 8 best third-placed teams — a reasonable v1 approximation
for title probabilities. Encoding the official R32 slotting is a future refinement.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def reconstruct_groups(fixtures: pd.DataFrame) -> list[list[str]]:
    adj = defaultdict(set)
    for r in fixtures.itertuples():
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        stack, comp = [t], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack += [y for y in adj[x] if y not in seen]
        comps.append(sorted(comp))
    return comps


# Official 2026 final-draw groups (normalised to dataset names), order A..L.
OFFICIAL_GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Qatar", "Bosnia and Herzegovina", "Switzerland"],
    "C": ["Scotland", "Brazil", "Haiti", "Morocco"],
    "D": ["Paraguay", "Turkey", "United States", "Australia"],
    "E": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "F": ["Tunisia", "Japan", "Netherlands", "Sweden"],
    "G": ["New Zealand", "Iran", "Egypt", "Belgium"],
    "H": ["Cape Verde", "Uruguay", "Spain", "Saudi Arabia"],
    "I": ["Senegal", "Norway", "France", "Iraq"],
    "J": ["Algeria", "Jordan", "Argentina", "Austria"],
    "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L": ["England", "Ghana", "Croatia", "Panama"],
}
_GI = {c: i for i, c in enumerate("ABCDEFGHIJKL")}

# Official Round-of-32 slot map (16 matches, bracket order). Each slot is
# ('W'|'RU', group) for winners/runners-up or ('3RD', eligible_groups) for the
# 8 third-place slots. Source: 2026 FIFA World Cup knockout-stage bracket.
_R32 = [
    (("RU", "A"), ("RU", "B")), (("W", "C"), ("RU", "F")),
    (("W", "E"), ("3RD", "ABCDF")), (("W", "F"), ("RU", "C")),
    (("RU", "E"), ("RU", "I")), (("W", "I"), ("3RD", "CDFGH")),
    (("W", "A"), ("3RD", "CEFHI")), (("W", "L"), ("3RD", "EHIJK")),
    (("W", "G"), ("3RD", "AEHIJ")), (("W", "D"), ("3RD", "BEFIJ")),
    (("W", "H"), ("RU", "J")), (("RU", "K"), ("RU", "L")),
    (("W", "B"), ("3RD", "EFGIJ")), (("RU", "D"), ("RU", "G")),
    (("W", "J"), ("RU", "H")), (("W", "K"), ("3RD", "DEIJL")),
]
# flattened third-slot positions (0..31) and their eligible group-index sets
_THIRD_POS, _THIRD_ELIG = [], []
for _mi, (_h, _a) in enumerate(_R32):
    for _si, _slot in enumerate((_h, _a)):
        if _slot[0] == "3RD":
            _THIRD_POS.append(_mi * 2 + _si)
            _THIRD_ELIG.append({_GI[c] for c in _slot[1]})


def _assign_thirds(qual: tuple) -> list:
    """Bijective assignment of the 8 qualifying third-place groups to the 8 third-slots,
    respecting each slot's eligible-group list (backtracking; memoised by caller)."""
    qs = set(qual)
    order = sorted(range(8), key=lambda s: len(_THIRD_ELIG[s] & qs))
    res = {}

    def bt(i, used):
        if i == len(order):
            return True
        slot = order[i]
        for g in _THIRD_ELIG[slot]:
            if g in qs and g not in used:
                res[slot] = g
                used.add(g)
                if bt(i + 1, used):
                    return True
                used.discard(g)
                del res[slot]
        return False

    if bt(0, set()):
        return [res[s] for s in range(8)]
    return list(qual)[:8]  # fallback (shouldn't happen for valid draws)


def _strength_arrays(model, teams: list[str]):
    atk = np.array([model.attack.get(t, 0.0) for t in teams])
    dfc = np.array([model.defence.get(t, 0.0) for t in teams])
    return atk, dfc


def _rank_desc(rng, *keys) -> np.ndarray:
    """Per-row descending rank order. keys = least→most significant before noise.
    Returns index order (n, k) with best first."""
    noise = rng.random(keys[0].shape)
    order = np.lexsort((noise, *keys), axis=1)  # ascending, primary = last key
    return order[:, ::-1]  # descending


def run(model, fixtures: pd.DataFrame, n: int = 50000, seed: int = 0,
        squads: dict | None = None, gb_topk: int = 12, context: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    fixture_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
    # use the official A..L draw when it matches the fixtures; else fall back
    if set(t for g in OFFICIAL_GROUPS.values() for t in g) == fixture_teams:
        groups = [OFFICIAL_GROUPS[c] for c in "ABCDEFGHIJKL"]
        official = True
    else:
        groups = reconstruct_groups(fixtures)
        official = False
    teams = sorted({t for g in groups for t in g})
    tid = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    atk, dfc = _strength_arrays(model, teams)
    inter = model.intercept
    hadv = model.home_adv

    pts = np.zeros((n, nt))
    gd = np.zeros((n, nt))
    gf = np.zeros((n, nt))

    context = context or {}
    for r in fixtures.itertuples():
        h, a = tid[r.home_team], tid[r.away_team]
        neutral = bool(r.neutral)
        ha = 0.0 if neutral else hadv
        cx = context.get(getattr(r, "fixture_id", None), {})
        lam = np.exp(inter + ha + atk[h] - dfc[a]) * cx.get("home_mult", 1.0)
        mu = np.exp(inter + atk[a] - dfc[h]) * cx.get("away_mult", 1.0)
        hg = rng.poisson(lam, n)
        ag = rng.poisson(mu, n)
        hw, aw, dr = hg > ag, ag > hg, hg == ag
        pts[:, h] += 3 * hw + dr
        pts[:, a] += 3 * aw + dr
        gd[:, h] += hg - ag
        gd[:, a] += ag - hg
        gf[:, h] += hg
        gf[:, a] += ag

    reached = {k: np.zeros(nt) for k in
               ("R32", "R16", "QF", "SF", "final", "champion")}
    group_adv = np.zeros(nt)  # top-2 finish probability

    # group standings → top 2 + collect third-placed
    top2_ids = np.empty((n, len(groups) * 2), dtype=int)
    third_ids = np.empty((n, len(groups)), dtype=int)
    third_pts = np.empty((n, len(groups)))
    third_gd = np.empty((n, len(groups)))
    third_gf = np.empty((n, len(groups)))

    for gi, g in enumerate(groups):
        ids = np.array([tid[t] for t in g])
        gp, ggd, ggf = pts[:, ids], gd[:, ids], gf[:, ids]
        order = _rank_desc(rng, ggf, ggd, gp)  # (n,4) best→worst, local idx
        glob = ids[order]  # (n,4) global team ids in finishing order
        top2_ids[:, gi * 2: gi * 2 + 2] = glob[:, :2]
        third_ids[:, gi] = glob[:, 2]
        rows = np.arange(n)
        third_pts[:, gi] = gp[rows, order[:, 2]]
        third_gd[:, gi] = ggd[rows, order[:, 2]]
        third_gf[:, gi] = ggf[rows, order[:, 2]]
        np.add.at(group_adv, glob[:, :2].ravel(), 1)

    # 8 best third-placed (group indices that qualify, per sim)
    torder = _rank_desc(rng, third_gf, third_gd, third_pts)  # (n,12)
    qual = np.sort(torder[:, :8], axis=1)  # (n,8) qualifying group indices

    win_ids = top2_ids[:, 0::2]   # (n,12) group winners, order A..L
    ru_ids = top2_ids[:, 1::2]    # (n,12) runners-up

    rows = np.arange(n)
    if official:
        # structured bracket: place teams into the official R32 slots
        cur = np.empty((n, 32), dtype=int)
        for mi, (h, a) in enumerate(_R32):
            for si, slot in enumerate((h, a)):
                pos = mi * 2 + si
                if slot[0] == "W":
                    cur[:, pos] = win_ids[:, _GI[slot[1]]]
                elif slot[0] == "RU":
                    cur[:, pos] = ru_ids[:, _GI[slot[1]]]
                # '3RD' slots filled below
        # assign qualifying thirds to the 8 third-slots (memoised by qualifying set)
        cache = {}
        slot_group = np.empty((n, len(_THIRD_POS)), dtype=int)
        qlist = qual.tolist()
        for i in range(n):
            key = tuple(qlist[i])
            a = cache.get(key)
            if a is None:
                a = cache[key] = _assign_thirds(key)
            slot_group[i] = a
        third_fill = np.take_along_axis(third_ids, slot_group, axis=1)  # (n,8)
        cur[:, _THIRD_POS] = third_fill
    else:
        third_adv = third_ids[rows[:, None], torder[:, :8]]
        advancers = np.concatenate([top2_ids, third_adv], axis=1)
        perm = rng.random(advancers.shape).argsort(axis=1)
        cur = np.take_along_axis(advancers, perm, axis=1)

    np.add.at(reached["R32"], cur.ravel(), 1)

    # total goals each team scores across the tournament (group + every KO match played)
    tour_goals = gf.copy()

    # knockout: structured bracket (cur is in official R32 slot order, adjacent-pair tree)
    stage_names = ["R16", "QF", "SF", "final", "champion"]
    for stage in stage_names:
        k = cur.shape[1]
        home = cur[:, 0:k:2]
        away = cur[:, 1:k:2]
        lam = np.exp(inter + atk[home] - dfc[away])
        mu = np.exp(inter + atk[away] - dfc[home])
        hg = rng.poisson(lam)
        ag = rng.poisson(mu)
        ri = np.repeat(np.arange(n), home.shape[1])
        np.add.at(tour_goals, (ri, home.ravel()), hg.ravel())
        np.add.at(tour_goals, (ri, away.ravel()), ag.ravel())
        p_home = lam / (lam + mu)
        coin = rng.random(hg.shape) < p_home
        home_wins = (hg > ag) | ((hg == ag) & coin)
        winners = np.where(home_wins, home, away)  # (n, k/2)
        np.add.at(reached[stage], winners.ravel(), 1)
        cur = winners

    # ---- Golden Boot: allocate each team's tournament goals to its players ----
    golden_boot = {}
    if squads:
        from ..model.players import goal_shares

        p_names, p_teams_idx, p_shares = [], [], []
        for ti, tm in enumerate(teams):
            sq = squads.get(tm)
            if not sq:
                continue
            shares = sorted(goal_shares(sq), key=lambda x: -x[2])[:gb_topk]
            for name, _pos, share in shares:
                p_names.append(name)
                p_teams_idx.append(ti)
                p_shares.append(share)
        if p_names:
            P = len(p_names)
            pg = np.zeros((n, P), dtype=np.int16)
            for j in range(P):
                lam = p_shares[j] * tour_goals[:, p_teams_idx[j]]
                pg[:, j] = rng.poisson(lam).astype(np.int16)
            winners = pg.argmax(axis=1)  # one Golden Boot winner per sim
            win_counts = np.bincount(winners, minlength=P)
            exp_goals = pg.mean(axis=0)
            order = np.argsort(win_counts)[::-1][:25]
            golden_boot = {
                "top_scorer_probability": {
                    f"{p_names[j]} ({teams[p_teams_idx[j]]})": round(float(win_counts[j]) / n, 4)
                    for j in order
                },
                "expected_goals": {
                    f"{p_names[j]} ({teams[p_teams_idx[j]]})": round(float(exp_goals[j]), 2)
                    for j in np.argsort(exp_goals)[::-1][:25]
                },
            }

    def table(counter):
        return {teams[i]: round(float(counter[i]) / n, 4) for i in range(nt)}

    champ = table(reached["champion"])
    out = {
        "n_simulations": n,
        "n_teams": nt,
        "n_groups": len(groups),
        "groups": {f"G{i+1}": g for i, g in enumerate(groups)},
        "title_probability": dict(sorted(champ.items(), key=lambda x: -x[1])),
        "advance_group_top2": dict(sorted(table(group_adv).items(), key=lambda x: -x[1])),
        "reach_knockout_R32": dict(sorted(table(reached["R32"]).items(), key=lambda x: -x[1])),
        "reach_quarterfinal": dict(sorted(table(reached["QF"]).items(), key=lambda x: -x[1])),
        "reach_final": dict(sorted(table(reached["final"]).items(), key=lambda x: -x[1])),
    }
    if golden_boot:
        out["golden_boot"] = golden_boot
    return out
