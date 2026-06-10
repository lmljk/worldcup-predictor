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

# 2026 group stage runs Jun 11-27; the R32 starts Jun 28. Rows after this date are
# knockout matches and must NEVER enter the group table.
_GROUP_END = pd.Timestamp("2026-06-27")


def _nm(s: str) -> str:
    """Accent-insensitive name key (same convention as players/injuries matching)."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def _shootout_winners() -> dict:
    """{(date, frozenset({a, b})): winner} from martj42 shootouts.csv (local, no network)."""
    from ..helpers import paths
    f = paths.HISTORICAL / "shootouts.csv"
    if not f.exists():
        return {}
    try:
        df = pd.read_csv(f)
        return {(pd.Timestamp(r.date).date(), frozenset((r.home_team, r.away_team))): r.winner
                for r in df.itertuples(index=False)}
    except Exception:   # noqa: BLE001 — a corrupt optional file must not kill the sim
        return {}

# Official Round-of-32 slot map — match order 73..88 (index 0..15). Each slot is
# ('W'|'RU', group) for winners/runners-up or ('3RD', eligible_groups) for the 8
# third-place slots. Source: 2026 FIFA World Cup official knockout bracket.
_R32 = [
    (("RU", "A"), ("RU", "B")),       # 73
    (("W", "E"), ("3RD", "ABCDF")),   # 74
    (("W", "F"), ("RU", "C")),        # 75
    (("W", "C"), ("RU", "F")),        # 76
    (("W", "I"), ("3RD", "CDFGH")),   # 77
    (("RU", "E"), ("RU", "I")),       # 78
    (("W", "A"), ("3RD", "CEFHI")),   # 79
    (("W", "L"), ("3RD", "EHIJK")),   # 80
    (("W", "D"), ("3RD", "BEFIJ")),   # 81
    (("W", "G"), ("3RD", "AEHIJ")),   # 82
    (("RU", "K"), ("RU", "L")),       # 83
    (("W", "H"), ("RU", "J")),        # 84
    (("W", "B"), ("3RD", "EFGIJ")),   # 85
    (("W", "J"), ("RU", "H")),        # 86
    (("W", "K"), ("3RD", "DEIJL")),   # 87
    (("RU", "D"), ("RU", "G")),       # 88
]
# Bracket tree (winners of these R32 match indices meet in each R16 match 89..96),
# then R16-winner indices into QF, etc. This is the real (non-sequential) adjacency.
_R16_PAIRS = [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
_QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]   # indices into the 8 R16 winners
_SF_PAIRS = [(0, 1), (2, 3)]                   # indices into the 4 QF winners
_F_PAIR = [(0, 1)]                             # indices into the 2 SF winners
# the 8 third-place slots: (R32 match index, eligible group-index set)
_THIRD_MATCHES = [(_mi, {_GI[c] for c in _a[1]})
                  for _mi, (_h, _a) in enumerate(_R32) if _a[0] == "3RD"]
_THIRD_ELIG = [e for _, e in _THIRD_MATCHES]


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
        squads: dict | None = None, gb_topk: int = 12, context: dict | None = None,
        scorer_goals: dict | None = None) -> dict:
    """`scorer_goals` (live mode): {(team, scorer_name): actual WC2026 goals so far} —
    real goals are credited to their real scorers; only *future* team goals are
    allocated by model share."""
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

    # ---- live conditioning: the sim must respect what has ALREADY happened ----
    # Group fixtures with a recorded score are pinned to the actual result in every
    # sim (no re-rolling reality); knockout rows (date after the group stage) are kept
    # out of the group table and pinned inside the bracket walk via result matrices.
    if official:
        group_fx = fixtures[fixtures["date"] <= _GROUP_END]
        ko_played = fixtures[(fixtures["date"] > _GROUP_END)
                             & fixtures["home_score"].notna()]
    else:
        group_fx, ko_played = fixtures, fixtures.iloc[0:0]
    actual_goals = np.zeros(nt)   # real goals already on the board, per team

    context = context or {}
    for r in group_fx.itertuples():
        h, a = tid[r.home_team], tid[r.away_team]
        hs = getattr(r, "home_score", None)
        if hs is not None and not pd.isna(hs):
            # played match → deterministic across all sims
            hg = np.full(n, int(hs), dtype=np.int64)
            ag = np.full(n, int(r.away_score), dtype=np.int64)
            actual_goals[h] += int(hs)
            actual_goals[a] += int(r.away_score)
        else:
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

    # played-knockout result matrices (both orientations): winner id, actual goals.
    # Inside _play, any simulated tie that reproduces a real played pairing is pinned.
    res_w = np.full((nt, nt), -1, dtype=np.int64)
    res_hg = np.full((nt, nt), -1, dtype=np.int64)
    res_ag = np.full((nt, nt), -1, dtype=np.int64)
    if len(ko_played):
        shoot = _shootout_winners()
        for r in ko_played.itertuples():
            h, a = tid[r.home_team], tid[r.away_team]
            hs, as_ = int(r.home_score), int(r.away_score)
            if hs > as_:
                w = h
            elif hs < as_:
                w = a
            else:
                wname = shoot.get((pd.Timestamp(r.date).date(),
                                   frozenset((r.home_team, r.away_team))))
                if wname is None or wname not in tid:
                    continue   # drawn KO with no shootout record yet → leave unpinned
                w = tid[wname]
            res_w[h, a] = res_w[a, h] = w
            res_hg[h, a], res_ag[h, a] = hs, as_
            res_hg[a, h], res_ag[a, h] = as_, hs
    ko_pinned = bool((res_w >= 0).any())

    reached = {k: np.zeros(nt) for k in
               ("R32", "R16", "QF", "SF", "final", "champion")}
    group_adv = np.zeros(nt)  # top-2 finish probability
    first_cnt = np.zeros(nt)  # win-group counter
    group_of = {}             # team -> group letter (official order)

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
        np.add.at(first_cnt, glob[:, 0], 1)
        for t in g:
            group_of[t] = "ABCDEFGHIJKL"[gi] if official else str(gi)

    # 8 best third-placed (group indices that qualify, per sim)
    torder = _rank_desc(rng, third_gf, third_gd, third_pts)  # (n,12)
    qual = np.sort(torder[:, :8], axis=1)  # (n,8) qualifying group indices

    win_ids = top2_ids[:, 0::2]   # (n,12) group winners, order A..L
    ru_ids = top2_ids[:, 1::2]    # (n,12) runners-up

    rows = np.arange(n)
    # total goals each team scores across the tournament (group + every KO match played)
    tour_goals = gf.copy()

    def _play(home, away):
        """One knockout round, vectorised: sample scorelines, resolve (penalties via a
        strength-weighted coin), accumulate goals, return winner team-ids (n, k).
        Pairings that were ACTUALLY played are pinned to the real score and winner."""
        lam = np.exp(inter + atk[home] - dfc[away])
        mu = np.exp(inter + atk[away] - dfc[home])
        hg = rng.poisson(lam)
        ag = rng.poisson(mu)
        if ko_pinned:
            ph, pa = res_hg[home, away], res_ag[home, away]
            hg = np.where(ph >= 0, ph, hg)
            ag = np.where(ph >= 0, pa, ag)
        ri = np.repeat(np.arange(n), home.shape[1])
        np.add.at(tour_goals, (ri, home.ravel()), hg.ravel())
        np.add.at(tour_goals, (ri, away.ravel()), ag.ravel())
        coin = rng.random(hg.shape) < (lam / (lam + mu))
        hw = (hg > ag) | ((hg == ag) & coin)
        win = np.where(hw, home, away)
        if ko_pinned:
            pw = res_w[home, away]
            win = np.where(pw >= 0, pw, win)
        return win

    if official:
        # build R32 home/away (n,16) from the official slot map (match order 73..88)
        home16 = np.empty((n, 16), dtype=int)
        away16 = np.empty((n, 16), dtype=int)
        for mi, (h, a) in enumerate(_R32):
            for slot, dst in ((h, home16), (a, away16)):
                if slot[0] == "W":
                    dst[:, mi] = win_ids[:, _GI[slot[1]]]
                elif slot[0] == "RU":
                    dst[:, mi] = ru_ids[:, _GI[slot[1]]]
        # qualifying thirds → the 8 eligible third-slots (memoised by qualifying set)
        cache = {}
        slot_group = np.empty((n, len(_THIRD_MATCHES)), dtype=int)
        qlist = qual.tolist()
        for i in range(n):
            key = tuple(qlist[i])
            asg = cache.get(key)
            if asg is None:
                asg = cache[key] = _assign_thirds(key)
            slot_group[i] = asg
        third_fill = np.take_along_axis(third_ids, slot_group, axis=1)  # (n,8)
        for j, (mi, _e) in enumerate(_THIRD_MATCHES):
            away16[:, mi] = third_fill[:, j]

        np.add.at(reached["R32"], home16.ravel(), 1)
        np.add.at(reached["R32"], away16.ravel(), 1)
        # walk the real bracket tree (NOT sequential pairs)
        w32 = _play(home16, away16)                                   # 16 → reach R16
        np.add.at(reached["R16"], w32.ravel(), 1)
        a, b = zip(*_R16_PAIRS)
        w16 = _play(w32[:, list(a)], w32[:, list(b)])                 # 8 → reach QF
        np.add.at(reached["QF"], w16.ravel(), 1)
        a, b = zip(*_QF_PAIRS)
        qf = _play(w16[:, list(a)], w16[:, list(b)])                  # 4 → reach SF
        np.add.at(reached["SF"], qf.ravel(), 1)
        a, b = zip(*_SF_PAIRS)
        sf = _play(qf[:, list(a)], qf[:, list(b)])                    # 2 → finalists
        np.add.at(reached["final"], sf.ravel(), 1)
        champ = _play(sf[:, [0]], sf[:, [1]])                         # champion
        np.add.at(reached["champion"], champ.ravel(), 1)
    else:
        # fallback (non-official draw): random single-elim bracket, sequential pairs
        third_adv = third_ids[rows[:, None], torder[:, :8]]
        advancers = np.concatenate([top2_ids, third_adv], axis=1)
        cur = np.take_along_axis(advancers, rng.random(advancers.shape).argsort(axis=1), axis=1)
        np.add.at(reached["R32"], cur.ravel(), 1)
        for stage in ("R16", "QF", "SF", "final", "champion"):
            k = cur.shape[1]
            cur = _play(cur[:, 0:k:2], cur[:, 1:k:2])
            np.add.at(reached[stage], cur.ravel(), 1)

    # ---- Golden Boot: allocate each team's tournament goals to its players ----
    golden_boot = {}
    if squads:
        from ..model.players import goal_shares

        # actual WC goals by (team, normalised scorer) → (display name, goals)
        sg: dict[tuple[str, str], tuple[str, int]] = {}
        for (tm_, nm_), g_ in (scorer_goals or {}).items():
            sg[(tm_, _nm(nm_))] = (nm_, int(g_))

        p_names, p_teams_idx, p_shares, p_seed = [], [], [], []
        for ti, tm in enumerate(teams):
            sq = squads.get(tm)
            if not sq:
                continue
            allsh = goal_shares(sq)
            picked = sorted(allsh, key=lambda x: -x[2])[:gb_topk]
            chosen = {name for name, _p, _s in picked}
            actual = {k[1]: v for k, v in sg.items() if k[0] == tm}
            if actual:
                # anyone who has REALLY scored at this WC must be on the board,
                # even if his model share didn't make the top-k cut
                for name, _pos, share in allsh:
                    if name not in chosen and _nm(name) in actual:
                        picked.append((name, _pos, share))
                        chosen.add(name)
            matched_keys = set()
            for name, _pos, share in picked:
                seed_g = actual.get(_nm(name), (None, 0))[1] if actual else 0
                if seed_g:
                    matched_keys.add(_nm(name))
                p_names.append(name)
                p_teams_idx.append(ti)
                p_shares.append(share)
                p_seed.append(seed_g)
            # real scorers whose name didn't match the squad list: keep their goals
            # (share 0 → seeded tally only, no future allocation)
            for nkey, (disp, g_) in actual.items():
                if nkey not in matched_keys:
                    p_names.append(disp)
                    p_teams_idx.append(ti)
                    p_shares.append(0.0)
                    p_seed.append(g_)
        if p_names:
            P = len(p_names)
            # only goals NOT yet on the board are allocated by share
            future_goals = np.clip(tour_goals - actual_goals[None, :], 0, None)
            pg = np.zeros((n, P), dtype=np.int16)
            for j in range(P):
                lam = p_shares[j] * future_goals[:, p_teams_idx[j]]
                pg[:, j] = (p_seed[j] + rng.poisson(lam)).astype(np.int16)
            winners = pg.argmax(axis=1)  # one Golden Boot winner per sim
            win_counts = np.bincount(winners, minlength=P)
            exp_goals = pg.mean(axis=0)
            order = np.argsort(win_counts)[::-1][:25]
            # the winner's tally per sim — the user-facing "Golden Boot = how many goals"
            # number (historical winners: 5-8). Distinct from each player's *marginal*
            # expectation, which averages over early exits and is naturally much lower.
            winner_tally = pg[np.arange(n), winners]
            tally_hist = np.bincount(winner_tally, minlength=10)
            golden_boot = {
                "top_scorer_probability": {
                    f"{p_names[j]} ({teams[p_teams_idx[j]]})": round(float(win_counts[j]) / n, 4)
                    for j in order
                },
                "expected_goals": {
                    f"{p_names[j]} ({teams[p_teams_idx[j]]})": round(float(exp_goals[j]), 2)
                    for j in np.argsort(exp_goals)[::-1][:25]
                },
                "winner_goals": {
                    "mean": round(float(winner_tally.mean()), 2),
                    "median": int(np.median(winner_tally)),
                    "p10": int(np.percentile(winner_tally, 10)),
                    "p90": int(np.percentile(winner_tally, 90)),
                    "distribution": {str(k): round(float(c) / n, 4)
                                     for k, c in enumerate(tally_hist) if c > 0},
                },
            }

    # qualify = top-2  OR  (3rd-placed AND that group is among the best-8 thirds)
    qualify_cnt = group_adv.copy()
    for gi in range(len(groups)):
        mask = (qual == gi).any(axis=1)
        np.add.at(qualify_cnt, third_ids[mask, gi], 1)

    def table(counter):
        return {teams[i]: round(float(counter[i]) / n, 4) for i in range(nt)}

    # per-group qualification standings (the "出线" view)
    group_standings = {}
    if official:
        for gi, letter in enumerate("ABCDEFGHIJKL"):
            tl = sorted(groups[gi],
                        key=lambda t: -(qualify_cnt[tid[t]]))
            group_standings[letter] = [{
                "team": t,
                "win_group": round(float(first_cnt[tid[t]]) / n, 4),
                "advance": round(float(group_adv[tid[t]]) / n, 4),
                "qualify": round(float(qualify_cnt[tid[t]]) / n, 4),
            } for t in tl]

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
        "group_standings": group_standings,
    }
    if golden_boot:
        out["golden_boot"] = golden_boot
    return out
