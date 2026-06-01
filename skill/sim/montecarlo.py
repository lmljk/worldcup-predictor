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
        squads: dict | None = None, gb_topk: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    groups = reconstruct_groups(fixtures)
    teams = sorted({t for g in groups for t in g})
    tid = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    atk, dfc = _strength_arrays(model, teams)
    inter = model.intercept
    hadv = model.home_adv

    pts = np.zeros((n, nt))
    gd = np.zeros((n, nt))
    gf = np.zeros((n, nt))

    for r in fixtures.itertuples():
        h, a = tid[r.home_team], tid[r.away_team]
        neutral = bool(r.neutral)
        ha = 0.0 if neutral else hadv
        lam = np.exp(inter + ha + atk[h] - dfc[a])
        mu = np.exp(inter + atk[a] - dfc[h])
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

    # 8 best third-placed
    torder = _rank_desc(rng, third_gf, third_gd, third_pts)  # (n,12)
    best8_local = torder[:, :8]
    rows = np.arange(n)[:, None]
    third_adv = third_ids[rows, best8_local]  # (n,8)

    advancers = np.concatenate([top2_ids, third_adv], axis=1)  # (n,32)
    np.add.at(reached["R32"], advancers.ravel(), 1)

    # total goals each team scores across the tournament (group + every KO match played)
    tour_goals = gf.copy()

    # knockout: random bracket per sim
    perm = rng.random(advancers.shape).argsort(axis=1)
    cur = np.take_along_axis(advancers, perm, axis=1)
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
