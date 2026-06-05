"""worldcup CLI.

Usage:
  python -m skill.helpers.cli fetch --all
  python -m skill.helpers.cli predict --all [--simulate]
  python -m skill.helpers.cli predict --match wc2026-000
  python -m skill.helpers.cli backtest --start 2010-01-01 --end 2026-05-31 [--xi 0.0019]
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import data_loader, paths

# Weight of the squad-talent prior on attack/defence (log-space nudge). Modest by design.
TALENT_WEIGHT = 0.10
# Weight of the live market in the per-match ensemble (market is hard to beat → high).
MARKET_WEIGHT = 0.60


def _cmd_fetch(args):
    summary = data_loader.fetch_all()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_predict(args):
    from ..model import dixon_coles as dc

    results = data_loader.load_results()
    fixtures = data_loader.load_wc2026_fixtures()
    as_of = pd.Timestamp.now().normalize()
    model = dc.fit(results, as_of=as_of)
    squads = data_loader.fetch_squads()

    # recent international form + penalty-taker flags (goalscorers dataset, free)
    try:
        from ..model.players import enrich_form
        squads = enrich_form(squads, data_loader.fetch_goalscorers(),
                             since=as_of - pd.Timedelta(days=730))
    except Exception as e:  # noqa: BLE001
        print(f"[form skipped] {e}", file=sys.stderr)

    # squad-talent prior (clubelo): nudge strength toward roster quality the
    # results-based model misses (e.g. France). Transparent, surfaced as a factor.
    talent, fc_team, fc_player = {}, {}, {}
    try:
        from ..model import fcratings, talent as talentmod
        talent = talentmod.squad_talent(squads, data_loader.fetch_club_elo())
        try:
            fc_team, fc_player = fcratings.team_ratings(squads)
            import re as _re
            import unicodedata as _u
            _nm = lambda s: _re.sub(r"[^a-z]", "", _u.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower())
            fc_lookup = {(t, _nm(n)): v for (t, n), v in fc_player.items()}
            for tm, ps in squads.items():
                for p in ps:
                    v = fc_lookup.get((tm, _nm(p["name"])))
                    if v:
                        p["fc_ovr"] = v["ovr"]
                        p["league"] = v.get("league") or p.get("league")
        except Exception as e:  # noqa: BLE001 — FC25 ratings optional
            print(f"[fc25 skipped] {e}", file=sys.stderr)
        model.attack, model.defence = talentmod.combined_adjust(
            model.attack, model.defence, talent, fc_team, weight=TALENT_WEIGHT)
    except Exception as e:  # noqa: BLE001 — talent is an optional enhancement
        print(f"[talent skipped] {e}", file=sys.stderr)

    from ..model import context as ctxmod
    ctx = ctxmod.compute(fixtures)

    # live per-match 1X2 market (Polymarket + Kalshi) — empty until books list matches,
    # then the ensemble blends it automatically.
    match_markets = data_loader.fetch_match_markets()
    if match_markets:
        print(f"[market] {len(match_markets)} live per-match markets found")

    if args.match:
        row = fixtures[fixtures["fixture_id"] == args.match]
        if row.empty:
            print(f"fixture {args.match} not found", file=sys.stderr)
            sys.exit(1)
        row = row.iloc[0]
        out = _predict_one(model, row, squads, ctx.get(row["fixture_id"]), match_markets)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    preds = []
    for _, row in fixtures.iterrows():
        try:
            preds.append(_predict_one(model, row, squads, ctx.get(row["fixture_id"]), match_markets))
        except Exception as e:  # noqa: BLE001 — keep going on sparse teams
            preds.append({"fixture_id": row["fixture_id"], "error": str(e)})
    rep = paths.report_dir() / "predictions.json"
    rep.write_text(json.dumps(preds, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {len(preds)} predictions -> {rep}")

    sim = None
    if args.simulate:
        from ..sim import montecarlo
        sim = montecarlo.run(model, fixtures, n=args.sims, squads=squads, context=ctx)
        (paths.report_dir() / "simulation.json").write_text(
            json.dumps(sim, indent=2, ensure_ascii=False)
        )
        top = list(sim["title_probability"].items())[:6]
        print(f"wrote tournament simulation ({args.sims} runs). Top title odds:")
        for t, p in top:
            print(f"  {t:<22} {p*100:5.1f}%")
        gb = sim.get("golden_boot", {}).get("top_scorer_probability", {})
        if gb:
            print("Golden Boot (top scorer prob):")
            for name, p in list(gb.items())[:6]:
                print(f"  {name:<34} {p*100:5.1f}%")

        # deterministic most-likely bracket (出线树) — single projected path to a champion
        from ..sim import bracket as bracketmod
        bk = bracketmod.project(model, fixtures)
        (paths.report_dir() / "bracket.json").write_text(
            json.dumps(bk, indent=2, ensure_ascii=False)
        )
        print(f"projected bracket champion: {bk['champion']}")

    # detail data for the dashboard's team/player search views
    tf, sq = _detail_payload(model, results, fixtures, squads, sim, talent, fc_team)
    (paths.report_dir() / "team_factors.json").write_text(json.dumps(tf, ensure_ascii=False))
    (paths.report_dir() / "squads_shares.json").write_text(json.dumps(sq, ensure_ascii=False))


def _detail_payload(model, results, fixtures, squads, sim, talent=None, fc_team=None):
    """Per-team model factors + per-player goal shares — powers the search detail views."""
    from ..model.elo import compute_elo_history
    from ..model.players import goal_shares, player_rate

    talent = talent or {}
    fc_team = fc_team or {}
    _, elo = compute_elo_history(results)
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    sim = sim or {}
    tp, adv = sim.get("title_probability", {}), sim.get("advance_group_top2", {})
    r32, fin = sim.get("reach_knockout_R32", {}), sim.get("reach_final", {})
    hosts = {"United States", "Canada", "Mexico"}
    tf = {}
    for t in teams:
        tl = talent.get(t, {})
        sqd = squads.get(t, [])
        team_ages = [p["age"] for p in sqd if p.get("age")]
        tf[t] = {
            "elo": round(float(elo.get(t, 1500)), 0),
            "attack": round(model.attack.get(t, 0.0), 3),
            "defence": round(model.defence.get(t, 0.0), 3),
            "talent": tl.get("talent"), "talent_z": tl.get("talent_z"),
            "fc_overall": fc_team.get(t, {}).get("fc_overall"),
            "fc_attack": fc_team.get(t, {}).get("fc_attack"),
            "fc_defence": fc_team.get(t, {}).get("fc_defence"),
            "avg_age": round(sum(team_ages) / len(team_ages), 1) if team_ages else None,
            "host": t in hosts,
            "title": tp.get(t), "advance": adv.get(t), "r32": r32.get(t), "final": fin.get(t),
        }
    sq = {}
    for t in teams:
        s = squads.get(t)
        if not s:
            continue
        share = {n: sh for n, _p, sh in goal_shares(s)}
        sq[t] = [{"name": p["name"], "pos": p["pos"], "caps": p["caps"], "goals": p["goals"],
                  "rate": round(player_rate(p), 3), "share": round(share.get(p["name"], 0.0), 4),
                  "recent_goals": p.get("recent_goals", 0), "pen_taker": bool(p.get("pen_taker")),
                  "age": p.get("age"), "fc_ovr": p.get("fc_ovr"), "league": p.get("league")}
                 for p in s]
    return tf, sq


def _predict_one(model, row, squads=None, ctx=None, match_markets=None) -> dict:
    import numpy as np

    from ..model import dixon_coles as dc
    from ..model import market as mkt
    from ..model.players import match_scorers

    home, away = row["home_team"], row["away_team"]
    neutral = bool(row["neutral"])
    if home not in model.attack or away not in model.attack:
        raise ValueError(f"insufficient data for {home} vs {away}")
    hm = ctx.get("home_mult", 1.0) if ctx else 1.0
    am = ctx.get("away_mult", 1.0) if ctx else 1.0
    mp = dc.match_probs(model, home, away, neutral, lam_mult=hm, mu_mult=am)
    mp["fixture_id"] = row["fixture_id"]
    mp["date"] = str(row["date"].date())
    mp["city"] = row.get("city")
    mp["country"] = row.get("country")
    if ctx and ctx.get("notes"):
        mp["context"] = ctx["notes"]
    # market-anchored ensemble: blend live 1X2 market into the model when available
    if match_markets and (home, away) in match_markets:
        p_mkt = np.array(match_markets[(home, away)], dtype=float)
        p_model = np.array([mp["p_home"], mp["p_draw"], mp["p_away"]])
        p_final = mkt.blend(p_mkt, p_model, w=MARKET_WEIGHT)
        mp["p_home"], mp["p_draw"], mp["p_away"] = (round(float(x), 4) for x in p_final)
        mp["market_1x2"] = [round(float(x), 4) for x in p_mkt]
        mp["edge"] = [round(float(x), 4) for x in (p_final - p_mkt)]
    if squads:
        lam_h, lam_a = model.lambdas(home, away, neutral)
        mp["scorers_home"] = match_scorers(lam_h * hm, squads.get(home, []), topn=3)
        mp["scorers_away"] = match_scorers(lam_a * am, squads.get(away, []), topn=3)
    # NOTE: market blend + context adjustments wired once live odds feed exists (M3/M5).
    return mp


def _slug(name: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _cmd_portraits(args):
    """Pre-download player photos from Wikipedia and self-host them in site/portraits/.

    Wikipedia/Wikimedia are blocked in mainland China, so client-side fetch fails there.
    We download server-side (proxy) and serve via our own Cloudflare tunnel instead.
    """
    import requests

    from ..model.players import goal_shares

    squads = data_loader.fetch_squads()
    # target: top-K goal-share players per team (covers match scorers + Golden Boot)
    names = []
    for t, ps in squads.items():
        top = sorted(goal_shares(ps), key=lambda x: -x[2])[: args.topk]
        names += [n for n, _p, _s in top]
    names = sorted(set(names))

    pdir = paths.SITE / "portraits"
    pdir.mkdir(exist_ok=True)
    pj = paths.DATA / "portraits.json"
    portraits = json.loads(pj.read_text()) if pj.exists() else {}
    # incremental: keep what's already on disk, only fetch the missing ones
    portraits = {n: f for n, f in portraits.items() if (pdir / f).exists()}
    todo = [n for n in names if not (pdir / (_slug(n) + ".jpg")).exists()]
    print(f"{len(names)} targets · {len(names) - len(todo)} already have photos · fetching {len(todo)}…")
    names = todo

    UA = {"User-Agent": "worldcup-predictor/0.1"}
    api = "https://en.wikipedia.org/w/api.php"
    url_map = {}
    for i in range(0, len(names), 50):
        batch = names[i:i + 50]
        try:
            j = requests.get(api, params={
                "action": "query", "titles": "|".join(batch), "prop": "pageimages",
                "piprop": "thumbnail", "pithumbsize": "240", "redirects": "1",
                "format": "json"}, headers=UA, timeout=30).json().get("query", {})
        except requests.RequestException:
            continue
        redir = {}
        for n in j.get("normalized", []):
            redir[n["from"]] = n["to"]
        for n in j.get("redirects", []):
            redir[n["from"]] = n["to"]

        def _final(t):
            seen = set()
            while t in redir and t not in seen:
                seen.add(t)
                t = redir[t]
            return t

        pages = {p.get("title"): p.get("thumbnail", {}).get("source")
                 for p in j.get("pages", {}).values()}
        for b in batch:
            u = pages.get(_final(b))
            if u:
                url_map[b] = u

    added = 0
    for name, u in url_map.items():
        fn = _slug(name) + ".jpg"
        try:
            img = requests.get(u, headers=UA, timeout=15)
            if img.status_code == 200 and img.content:
                (pdir / fn).write_bytes(img.content)
                portraits[name] = fn
                added += 1
        except requests.RequestException:
            continue
    pj.write_text(json.dumps(portraits, ensure_ascii=False))
    print(f"added {added} portraits · total {len(portraits)} -> {pdir}")


def _cmd_players(args):
    """Per-match likely scorers for a fixture (or refresh squads with --refresh)."""
    from ..model import dixon_coles as dc

    if args.refresh:
        sq = data_loader.fetch_squads(force=True)
        print(f"refreshed squads: {len(sq)} teams, {sum(len(v) for v in sq.values())} players")
        return
    results = data_loader.load_results()
    fixtures = data_loader.load_wc2026_fixtures()
    squads = data_loader.fetch_squads()
    model = dc.fit(results, as_of=pd.Timestamp.now().normalize())
    row = fixtures[fixtures["fixture_id"] == args.match]
    if row.empty:
        print(f"fixture {args.match} not found", file=sys.stderr)
        sys.exit(1)
    out = _predict_one(model, row.iloc[0], squads)
    h, a = out["home"], out["away"]
    print(f"{h} (λ={out['lambda_home']}) vs {a} (λ={out['lambda_away']})  [{out['date']}]\n")
    for side, key in ((h, "scorers_home"), (a, "scorers_away")):
        print(f"  {side} — likely scorers:")
        for s in out.get(key, []):
            print(f"    {s['name']:<24}{s['pos']}  P(score) {s['p_score']*100:4.1f}%  xG {s['exp_goals']}")
        print()


def _cmd_market(args):
    """Print our Monte Carlo title odds next to Polymarket's de-vigged market, with edge."""
    rep = paths.report_dir(args.date)
    sim_f = rep / "simulation.json"
    model_title = json.loads(sim_f.read_text()).get("title_probability", {}) if sim_f.exists() else {}
    mk = data_loader.fetch_polymarket_winner(team_filter=set(model_title) or None)
    if "error" in mk:
        print(f"polymarket fetch failed: {mk['error']}", file=sys.stderr)
        sys.exit(1)
    market = mk["implied_title_prob"]
    print(f"Polymarket 'World Cup Winner' — overround {mk['overround']}, {mk['n_teams']} teams\n")
    print(f"{'Team':<20}{'Model':>8}{'Market':>8}{'Edge':>8}")
    teams = sorted(set(market) | set(model_title), key=lambda t: -market.get(t, 0))
    for t in teams[:20]:
        m, k = model_title.get(t, 0), market.get(t, 0)
        print(f"{t:<20}{m*100:7.1f}%{k*100:7.1f}%{(m-k)*100:+7.1f}%")


def _cmd_review(args):
    """Daily live loop: refresh data, score completed matches vs our prior predictions,
    re-predict upcoming fixtures, re-simulate, and republish the dashboard."""
    import glob

    from ..backtest import metrics

    data_loader.fetch_historical(force=True)  # pull latest scores
    data_loader.fetch_fd_matches(force=True)  # refresh official schedule + live scores/knockout teams
    hist = data_loader.fetch_historical()
    played = hist[
        (hist["tournament"] == paths.WC2026_TOURNAMENT)
        & (hist["date"] >= pd.Timestamp(paths.WC2026_START))
        & hist["home_score"].notna()
    ].copy()

    # index our past predictions by (date, home, away) -> probs
    pred_index = {}
    for f in sorted(glob.glob(str(paths.REPORTS / "*" / "predictions.json"))):
        for p in json.loads(open(f).read()):
            if p.get("error"):
                continue
            key = (p.get("date"), p.get("home"), p.get("away"))
            pred_index.setdefault(key, p)  # earliest prediction wins

    scored = []
    for r in played.itertuples():
        key = (str(r.date.date()), r.home_team, r.away_team)
        p = pred_index.get(key)
        if not p:
            continue
        probs = np.array([p["p_home"], p["p_draw"], p["p_away"]])
        outcome = metrics.outcome_index(int(r.home_score), int(r.away_score))
        scored.append({
            "date": key[0], "match": f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}",
            "rps": round(metrics.rps(probs, outcome), 4),
            "log_loss": round(metrics.log_loss(probs, outcome), 4),
            "hit": bool(int(np.argmax(probs)) == outcome),
        })

    live = {"matches_scored": len(scored)}
    if scored:
        live["mean_rps"] = round(float(np.mean([s["rps"] for s in scored])), 4)
        live["hit_rate"] = round(float(np.mean([s["hit"] for s in scored])), 4)
    review = {"reviewed_at": pd.Timestamp.now().isoformat(timespec="minutes"),
              "live_calibration": live, "detail": scored}
    (paths.report_dir() / "review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False))
    print(json.dumps(live, indent=2, ensure_ascii=False))

    # refresh forward-looking predictions + sim + publish
    _cmd_predict(argparse.Namespace(match=None, all=True, simulate=True, sims=args.sims))
    _cmd_publish(argparse.Namespace(date=None))


def _schedule_payload(preds: list) -> list:
    """Official 104-match schedule (football-data.org) in chronological order, every stage,
    joined to our predictions + results. Knockout matches show their round until teams are
    drawn, then fill in with a prediction and (once played) a ✓/✗ vs the actual result."""
    import re
    import unicodedata

    def nm(s):
        s = unicodedata.normalize("NFD", str(s or "")).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z]", "", s.lower())

    pidx = {(nm(p["home"]), nm(p["away"])): p for p in preds if not p.get("error")}
    out = []
    for m in sorted(data_loader.fetch_fd_matches(), key=lambda x: x.get("utcDate", "")):
        home = data_loader.fd_canon((m.get("homeTeam") or {}).get("name"))
        away = data_loader.fd_canon((m.get("awayTeam") or {}).get("name"))
        ft = (m.get("score") or {}).get("fullTime") or {}
        status = m.get("status")
        score = (f"{ft['home']}-{ft['away']}"
                 if status == "FINISHED" and ft.get("home") is not None else None)
        rec = {"date": (m.get("utcDate") or "")[:10], "stage": m.get("stage"),
               "matchday": m.get("matchday"), "group": (m.get("group") or "").replace("GROUP_", ""),
               "home": home, "away": away, "status": status, "score": score}
        p = pidx.get((nm(home), nm(away))) if home and away else None
        if p:
            rec["probs"] = [p["p_home"], p["p_draw"], p["p_away"]]
        elif home and away and (nm(away), nm(home)) in pidx:  # orientation flipped
            q = pidx[(nm(away), nm(home))]
            rec["probs"] = [q["p_away"], q["p_draw"], q["p_home"]]  # swap home/away
            p = q
        if rec.get("probs"):
            rec["pick"] = int(max(range(3), key=lambda i: rec["probs"][i]))
        if p and score:
            h, a = (int(x) for x in score.split("-"))
            outcome = 0 if h > a else (1 if h == a else 2)
            rec["correct"] = rec["pick"] == outcome
        out.append(rec)
    return out


def _accuracy_payload() -> dict:
    """Live prediction scoreboard: every completed WC match vs our last pre-kickoff forecast.

    For each played WC match we take the prediction from the most recent report dated on or
    before the match date (no hindsight), mark the 1X2 pick correct/wrong, and aggregate an
    overall accuracy-to-date (hit rate, RPS, Brier). Empty until the tournament starts.
    """
    import glob

    from ..backtest import metrics

    res = data_loader.fetch_historical()
    played = res[(res["tournament"] == paths.WC2026_TOURNAMENT)
                 & (res["date"] >= pd.Timestamp(paths.WC2026_START))
                 & res["home_score"].notna()].copy()
    # index predictions by (matchdate, home, away) -> keep latest report date <= matchdate
    idx = {}
    for f in sorted(glob.glob(str(paths.REPORTS / "*" / "predictions.json"))):
        rdate = f.split("/")[-2]
        for p in json.loads(open(f).read()):
            if p.get("error") or not p.get("date"):
                continue
            if rdate <= p["date"]:  # forecast made on/before kickoff day
                key = (p["date"], p["home"], p["away"])
                if key not in idx or rdate >= idx[key][0]:
                    idx[key] = (rdate, p)

    labels = lambda h, a: [f"{h} win", "Draw", f"{a} win"]
    by_match, rows = {}, []
    for r in played.itertuples():
        key = (str(r.date.date()), r.home_team, r.away_team)
        hit = idx.get(key)
        if not hit:
            continue
        p = hit[1]
        probs = np.array([p["p_home"], p["p_draw"], p["p_away"]])
        pick = int(np.argmax(probs))
        outcome = metrics.outcome_index(int(r.home_score), int(r.away_score))
        correct = pick == outcome
        rec = {
            "date": key[0], "home": r.home_team, "away": r.away_team,
            "score": f"{int(r.home_score)}-{int(r.away_score)}",
            "pick": labels(r.home_team, r.away_team)[pick],
            "pick_prob": round(float(probs[pick]), 3),
            "actual": labels(r.home_team, r.away_team)[outcome],
            "correct": bool(correct),
            "rps": round(metrics.rps(probs, outcome), 4),
        }
        rows.append(rec)
        by_match[f"{key[0]}|{r.home_team}|{r.away_team}"] = {
            "score": rec["score"], "correct": rec["correct"], "actual_idx": outcome}
    rows.sort(key=lambda x: x["date"])
    summary = {"n": len(rows)}
    if rows:
        summary["correct"] = sum(1 for x in rows if x["correct"])
        summary["hit_rate"] = round(summary["correct"] / len(rows), 4)
        summary["mean_rps"] = round(float(np.mean([x["rps"] for x in rows])), 4)
    return {"summary": summary, "results": rows, "by_match": by_match}


def _backtest_headline() -> dict:
    """Headline credibility numbers from the most recent walk-forward backtest.

    Surfaces the statistical backbone's validated metrics for the dashboard's
    methodology card. ELO is the stronger 1X2 baseline (see reports/backtests/
    FINDINGS.md); we report it as a conservative floor for the shipped ensemble.
    Returns {} if no backtest has been run yet.
    """
    import glob

    files = glob.glob(str(paths.REPORTS / "backtests" / "backtest_*.json"))
    runs = []
    for f in files:
        try:
            runs.append(json.loads(open(f).read()))
        except Exception:  # noqa: BLE001
            continue
    if not runs:
        return {}
    # prefer the largest-sample backtest — the most robust headline, not the newest file
    bt = max(runs, key=lambda r: (r.get("elo_baseline", {}) or r.get("dixon_coles", {})).get("n", 0))
    elo, dc = bt.get("elo_baseline", {}), bt.get("dixon_coles", {})
    base = elo if elo.get("top_pick_accuracy", 0) >= dc.get("top_pick_accuracy", 0) else dc
    win = bt.get("window") or []
    return {
        "top_pick": base.get("top_pick_accuracy"),
        "rps": base.get("mean_rps"),
        "n": base.get("n"),
        "window": f"{win[0][:4]}–{win[1][:4]}" if len(win) == 2 else "",
    }


def _cmd_publish(args):
    """Bundle the latest predictions + simulation into site/data.json for the dashboard."""
    import datetime as _dt

    rep = paths.report_dir(args.date)
    preds_f = rep / "predictions.json"
    sim_f = rep / "simulation.json"
    if not preds_f.exists():
        print(f"no predictions at {preds_f} — run `predict --all --simulate` first", file=sys.stderr)
        sys.exit(1)
    sim = json.loads(sim_f.read_text()) if sim_f.exists() else {}
    bracket_f = rep / "bracket.json"
    tf_f, sq_f = rep / "team_factors.json", rep / "squads_shares.json"
    data = {
        "generated_at": _dt.datetime.now().isoformat(timespec="minutes"),
        "report_date": rep.name,
        "predictions": json.loads(preds_f.read_text()),
        "simulation": sim,
        "bracket": json.loads(bracket_f.read_text()) if bracket_f.exists() else {},
        "team_factors": json.loads(tf_f.read_text()) if tf_f.exists() else {},
        "squads": json.loads(sq_f.read_text()) if sq_f.exists() else {},
        "portraits": json.loads((paths.DATA / "portraits.json").read_text())
        if (paths.DATA / "portraits.json").exists() else {},
        "live_accuracy": _accuracy_payload(),
        "backtest": _backtest_headline(),
        "schedule": _schedule_payload(json.loads(preds_f.read_text())),
    }
    # Market anchor: Polymarket title odds vs our Monte Carlo (model-vs-market + edge).
    model_title = sim.get("title_probability", {})
    if model_title:
        teams = set(model_title)
        mk = data_loader.fetch_polymarket_winner(team_filter=teams)
        if "implied_title_prob" in mk:
            market = dict(mk["implied_title_prob"])
            # multi-market consensus: average every source that quotes a team, then renormalise
            extra = {"polymarket": market}
            kal = data_loader.fetch_kalshi_title(team_filter=teams)
            if kal:
                extra["kalshi"] = kal
            oa = data_loader.fetch_oddsapi_title(team_filter=teams)
            if oa:
                extra["the_odds_api"] = oa  # 50+ sportsbooks incl. Pinnacle (paid key)
            if len(extra) > 1:
                allt = set().union(*[set(s) for s in extra.values()])
                merged = {t: sum(s[t] for s in extra.values() if t in s)
                             / sum(1 for s in extra.values() if t in s) for t in allt}
                s = sum(merged.values()) or 1
                market = {t: round(v / s, 5) for t, v in merged.items()}
                mk["sources"] = "+".join(extra)
                mk["n_sources"] = len(extra)
                mk["implied_title_prob"] = dict(sorted(market.items(), key=lambda x: -x[1]))
            comparison = sorted(
                ({"team": t, "model": round(model_title.get(t, 0), 4),
                  "market": round(market.get(t, 0), 4),
                  "edge": round(model_title.get(t, 0) - market.get(t, 0), 4)}
                 for t in teams | set(market)),
                key=lambda x: -x["market"],
            )
            data["market_title"] = mk
            data["title_comparison"] = comparison

    # ---- probability movement tracking (odds drift = the public footprint of insider info) ----
    # Snapshot today's model + market title probs; movement is computed vs ~7 entries ago.
    hist_f = paths.DATA / "history.json"
    hist = json.loads(hist_f.read_text()) if hist_f.exists() else {}
    mkt_now = (data.get("market_title", {}) or {}).get("implied_title_prob", {})
    hist[data["report_date"]] = {"model": model_title, "market": mkt_now}
    hist = dict(sorted(hist.items())[-90:])  # keep last 90 days
    hist_f.write_text(json.dumps(hist, ensure_ascii=False))
    dates = sorted(hist)
    if len(dates) >= 2:
        prev = hist[dates[max(0, len(dates) - 8)]]  # ~7 snapshots ago
        cur = hist[dates[-1]]
        teams_m = set(cur["market"]) | set(prev.get("market", {}))
        movers = []
        for t in teams_m:
            mc, mp = cur["market"].get(t), prev.get("market", {}).get(t)
            if mc is None or mp is None:
                continue
            movers.append({"team": t, "market_now": round(mc, 4),
                           "delta": round(mc - mp, 4),
                           "model_now": round(cur["model"].get(t, 0), 4)})
        movers.sort(key=lambda x: -abs(x["delta"]))
        # per-team market series (for sparklines), last 14 snapshots
        series = {t: [round(hist[d]["market"].get(t, 0), 4) for d in dates[-14:]]
                  for t in [m["team"] for m in movers[:20]]}
        data["movement"] = {"since": dates[max(0, len(dates) - 8)], "as_of": dates[-1],
                            "top_movers": movers[:20], "market_series": series}
    (paths.SITE / "data.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"published -> {paths.SITE / 'data.json'} "
          f"({len(data['predictions'])} fixtures, sim={'yes' if sim_f.exists() else 'no'}, "
          f"market={'yes' if data.get('market_title') else 'no'})")


def _cmd_backtest(args):
    from ..backtest import walkforward

    results = data_loader.load_results()
    out = walkforward.run(
        results, start=args.start, end=args.end,
        refit_days=args.refit_days, majors_only=not args.all_matches,
        xi=args.xi, verbose=False,
    )
    fname = f"backtest_{args.start}_{args.end}_xi{args.xi}.json"
    (paths.BACKTESTS / fname).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="worldcup")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch")
    pf.add_argument("--all", action="store_true")
    pf.set_defaults(func=_cmd_fetch)

    pp = sub.add_parser("predict")
    pp.add_argument("--match", default=None)
    pp.add_argument("--all", action="store_true")
    pp.add_argument("--simulate", action="store_true")
    pp.add_argument("--sims", type=int, default=50000)
    pp.set_defaults(func=_cmd_predict)

    pu = sub.add_parser("publish")
    pu.add_argument("--date", default=None)
    pu.set_defaults(func=_cmd_publish)

    pr = sub.add_parser("review")
    pr.add_argument("--sims", type=int, default=50000)
    pr.set_defaults(func=_cmd_review)

    pm = sub.add_parser("market")
    pm.add_argument("--date", default=None)
    pm.set_defaults(func=_cmd_market)

    pl = sub.add_parser("players")
    pl.add_argument("--match", default=None)
    pl.add_argument("--refresh", action="store_true")
    pl.set_defaults(func=_cmd_players)

    pp2 = sub.add_parser("portraits")
    pp2.add_argument("--topk", type=int, default=10)
    pp2.set_defaults(func=_cmd_portraits)

    pb = sub.add_parser("backtest")
    pb.add_argument("--start", default="2010-01-01")
    pb.add_argument("--end", default="2026-05-31")
    pb.add_argument("--xi", type=float, default=0.0010)
    pb.add_argument("--refit-days", type=int, default=60, dest="refit_days")
    pb.add_argument("--all-matches", action="store_true", dest="all_matches")
    pb.set_defaults(func=_cmd_backtest)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
