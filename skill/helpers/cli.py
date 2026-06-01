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


def _cmd_fetch(args):
    summary = data_loader.fetch_all()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_predict(args):
    from ..model import dixon_coles as dc

    results = data_loader.load_results()
    fixtures = data_loader.load_wc2026_fixtures()
    as_of = pd.Timestamp.now().normalize()
    model = dc.fit(results, as_of=as_of)

    if args.match:
        row = fixtures[fixtures["fixture_id"] == args.match]
        if row.empty:
            print(f"fixture {args.match} not found", file=sys.stderr)
            sys.exit(1)
        row = row.iloc[0]
        out = _predict_one(model, row)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    preds = []
    for _, row in fixtures.iterrows():
        try:
            preds.append(_predict_one(model, row))
        except Exception as e:  # noqa: BLE001 — keep going on sparse teams
            preds.append({"fixture_id": row["fixture_id"], "error": str(e)})
    rep = paths.report_dir() / "predictions.json"
    rep.write_text(json.dumps(preds, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {len(preds)} predictions -> {rep}")

    if args.simulate:
        from ..sim import montecarlo
        sim = montecarlo.run(model, fixtures, n=args.sims)
        (paths.report_dir() / "simulation.json").write_text(
            json.dumps(sim, indent=2, ensure_ascii=False)
        )
        top = list(sim["title_probability"].items())[:8]
        print(f"wrote tournament simulation ({args.sims} runs). Top title odds:")
        for t, p in top:
            print(f"  {t:<20} {p*100:5.1f}%")


def _predict_one(model, row) -> dict:
    from ..model import dixon_coles as dc

    home, away = row["home_team"], row["away_team"]
    neutral = bool(row["neutral"])
    if home not in model.attack or away not in model.attack:
        raise ValueError(f"insufficient data for {home} vs {away}")
    mp = dc.match_probs(model, home, away, neutral)
    mp["fixture_id"] = row["fixture_id"]
    mp["date"] = str(row["date"].date())
    mp["city"] = row.get("city")
    mp["country"] = row.get("country")
    # NOTE: market blend + context adjustments wired once live odds feed exists (M3/M5).
    return mp


def _cmd_review(args):
    """Daily live loop: refresh data, score completed matches vs our prior predictions,
    re-predict upcoming fixtures, re-simulate, and republish the dashboard."""
    import glob

    from ..backtest import metrics

    data_loader.fetch_historical(force=True)  # pull latest scores
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


def _cmd_publish(args):
    """Bundle the latest predictions + simulation into site/data.json for the dashboard."""
    import datetime as _dt

    rep = paths.report_dir(args.date)
    preds_f = rep / "predictions.json"
    sim_f = rep / "simulation.json"
    if not preds_f.exists():
        print(f"no predictions at {preds_f} — run `predict --all --simulate` first", file=sys.stderr)
        sys.exit(1)
    data = {
        "generated_at": _dt.datetime.now().isoformat(timespec="minutes"),
        "report_date": rep.name,
        "predictions": json.loads(preds_f.read_text()),
        "simulation": json.loads(sim_f.read_text()) if sim_f.exists() else {},
    }
    (paths.SITE / "data.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"published -> {paths.SITE / 'data.json'} "
          f"({len(data['predictions'])} fixtures, sim={'yes' if sim_f.exists() else 'no'})")


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
