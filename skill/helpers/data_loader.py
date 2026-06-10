"""Data acquisition: historical results, WC2026 fixtures, weather, Polymarket.

All sources here are public and need no auth. Everything caches to disk so backtests
are reproducible and we never hammer an API.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from . import paths

paths.load_dotenv()

UA = {"User-Agent": "worldcup-predictor/0.1 (+https://github.com/rrclaw)"}


def _refresh_csv(url: str, dest, required: set, min_rows: int) -> None:
    """Download → VALIDATE in memory → atomic replace. A truncated/garbled download
    must never overwrite a good cache (the live loop force-refetches twice a day;
    one bad pull would otherwise brick the whole pipeline). On failure: keep the old
    cache and warn; raise only if there is no cache at all to fall back to."""
    import io
    import os
    import sys as _sys
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        if not required.issubset(df.columns) or len(df) < min_rows:
            raise ValueError(f"validation failed: cols={list(df.columns)[:6]} rows={len(df)}")
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(r.content)
        os.replace(tmp, dest)
    except Exception as e:  # noqa: BLE001 — degrade to cache, never corrupt it
        if dest.exists():
            print(f"[refresh failed, keeping cached {dest.name}] {e}", file=_sys.stderr)
        else:
            raise


def _write_json_atomic(path, text: str) -> None:
    """Partial-write-safe JSON cache write (tmp + rename)."""
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def fetch_historical(force: bool = False) -> pd.DataFrame:
    """martj42 international results (1872-now) + WC2026 fixtures in one CSV."""
    csv = paths.HISTORICAL_RESULTS_CSV
    if force or not csv.exists():
        _refresh_csv(paths.MARTJ42_RESULTS_URL, csv, min_rows=10000,
                     required={"date", "home_team", "away_team", "tournament"})
    df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def load_results(played_only: bool = True) -> pd.DataFrame:
    """Historical matches with known scores (for fitting models)."""
    df = fetch_historical()
    df = df.dropna(subset=["date"])
    if played_only:
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


MARTJ42_GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)
GOALSCORERS_CSV = paths.HISTORICAL / "goalscorers.csv"


def fetch_goalscorers(force: bool = False) -> pd.DataFrame:
    """martj42 goalscorers (date, team, scorer, penalty, own_goal) — free, no auth.
    Powers recent-form and penalty-taker signals for the player model."""
    if force or not GOALSCORERS_CSV.exists():
        _refresh_csv(MARTJ42_GOALSCORERS_URL, GOALSCORERS_CSV, min_rows=10000,
                     required={"date", "team", "scorer"})
    df = pd.read_csv(GOALSCORERS_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("penalty", "own_goal"):
        df[c] = df[c].astype(str).str.upper().eq("TRUE")
    return df.dropna(subset=["date", "scorer"])


def load_wc2026_fixtures() -> pd.DataFrame:
    """Future WC2026 rows (scores still NA) = the fixture list to predict."""
    df = fetch_historical()
    mask = (
        (df["tournament"] == paths.WC2026_TOURNAMENT)
        & (df["date"] >= pd.Timestamp(paths.WC2026_START))
    )
    fx = df.loc[mask].copy().sort_values("date").reset_index(drop=True)
    fx["fixture_id"] = [f"wc2026-{i:03d}" for i in range(len(fx))]
    return fx


def load_injuries() -> list[dict]:
    """Curated tournament-long absentees (transparent injury prior). Empty until
    confirmed rulings are added to data/injuries_wc2026.json."""
    f = paths.DATA / "injuries_wc2026.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text()).get("out", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def fetch_weather(lat: float, lon: float, when: str) -> dict[str, Any]:
    """Open-Meteo forecast/hindcast for a venue at a date (no key)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "start_date": when,
        "end_date": when,
        "timezone": "auto",
    }
    try:
        r = requests.get(paths.OPEN_METEO_FORECAST, params=params, headers=UA, timeout=30)
        r.raise_for_status()
        return r.json().get("daily", {})
    except requests.RequestException as e:
        return {"error": str(e)}


def _canon(team: str) -> str:
    """All sources share one authoritative alias table (helpers/teamnames.py)."""
    from .teamnames import canon
    return canon(team)


def fetch_polymarket_winner(team_filter: set[str] | None = None) -> dict[str, Any]:
    """Polymarket 'World Cup Winner' market → de-vigged implied title probabilities.

    Each sub-market is 'Will <team> win the 2026 FIFA World Cup?' (Yes price ≈ prob).
    We map names to our dataset, keep only real WC teams, and normalise so the
    de-vigged probabilities sum to 1 (removes the ~3% market overround).
    """
    import re

    try:
        ev = requests.get("https://gamma-api.polymarket.com/events/slug/world-cup-winner",
                          headers=UA, timeout=30)
        if ev.status_code != 200:
            ev = requests.get("https://gamma-api.polymarket.com/events/30615", headers=UA, timeout=30)
        ev.raise_for_status()
        data = ev.json()
    except requests.RequestException as e:
        return {"error": str(e)}

    raw = {}
    for m in data.get("markets", []):
        mt = re.match(r"Will (.+?) win the 2026 FIFA World Cup\?", m.get("question", ""))
        pr = m.get("outcomePrices")
        if not mt or not pr:
            continue
        try:
            yes = float(json.loads(pr)[0])
        except (ValueError, json.JSONDecodeError, IndexError):
            continue
        team = _canon(mt.group(1))
        if team_filter and team not in team_filter:
            continue
        raw[team] = yes

    overround = sum(raw.values())
    devig = {t: round(p / overround, 5) for t, p in raw.items()} if overround else {}
    return {
        "source": "polymarket:world-cup-winner",
        "fetched_at": datetime.now().isoformat(timespec="minutes"),
        "overround": round(overround, 4),
        "n_teams": len(devig),
        "implied_title_prob": dict(sorted(devig.items(), key=lambda x: -x[1])),
    }


def fetch_polymarket(query: str = "World Cup", limit: int = 100) -> list[dict[str, Any]]:
    """Polymarket Gamma markets matching a query (public, no key)."""
    try:
        r = requests.get(
            f"{paths.POLYMARKET_GAMMA}/markets",
            params={"closed": "false", "limit": limit, "search": query},
            headers=UA,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("data", [])
    except requests.RequestException as e:
        return [{"error": str(e)}]


WIKI_SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
SQUADS_JSON = paths.DATA / "squads_wc2026.json"
# Wikipedia heading names -> our dataset names (where they differ).
def _wiki_team(name: str) -> str:
    return _canon(name)


def fetch_squads(force: bool = False, team_filter: set[str] | None = None) -> dict[str, list[dict]]:
    """Scrape the Wikipedia '2026 FIFA World Cup squads' page.

    Returns {team: [{name, pos, caps, goals}]}. Free, no key. Career international
    'goals'/'caps' drive each player's scoring rate. Cached to disk; rosters change
    (injury replacements) so pass force=True to refresh.
    """
    import json as _json
    import re

    from bs4 import BeautifulSoup

    if SQUADS_JSON.exists() and not force:
        data = _json.loads(SQUADS_JSON.read_text())
    else:
        html = requests.get(WIKI_SQUADS_URL, headers=UA, timeout=45).text
        soup = BeautifulSoup(html, "lxml")
        data = {}
        for tb in soup.select("table.wikitable"):
            head = [th.get_text(strip=True) for th in tb.select("tr th")][:7]
            if not any(h == "Caps" for h in head):
                continue
            h = tb.find_previous(["h2", "h3", "h4"])
            if not h:
                continue
            hl = h.find(class_="mw-headline")
            raw = (hl.get_text(strip=True) if hl else h.get_text(strip=True))
            raw = re.sub(r"\[edit\]\s*$", "", raw).strip()
            team = _wiki_team(raw)
            if not team:
                continue
            players = []
            for r in tb.select("tr")[1:]:
                cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
                if len(cells) < 6:
                    continue
                pos = (cells[1].split()[-1] if cells[1] else "").upper()
                name = re.sub(r"\s*\(.*?\)\s*$", "", cells[2]).strip()  # drop (c) etc.
                caps = re.sub(r"[^\d]", "", cells[4]) or "0"
                goals = re.sub(r"[^\d]", "", cells[5]) or "0"
                club = cells[6].strip() if len(cells) > 6 else ""
                if pos not in {"GK", "DF", "MF", "FW"} or not name:
                    continue
                dob = ""
                age = None
                m = re.search(r"(\d{4}-\d{2}-\d{2})", cells[3]) if len(cells) > 3 else None
                if m:
                    dob = m.group(1)
                    by, bm, bd = (int(x) for x in dob.split("-"))
                    age = paths.WC2026_START.year - by - ((6, 11) < (bm, bd))
                players.append({"name": name, "pos": pos, "caps": int(caps),
                                "goals": int(goals), "club": club, "dob": dob, "age": age})
            if players:
                data[team] = players
        _write_json_atomic(SQUADS_JSON, _json.dumps(data, ensure_ascii=False, indent=1))

    if team_filter:
        return {t: p for t, p in data.items() if t in team_filter}
    return data


CLUBELO_URL = "http://api.clubelo.com/{date}"
CLUB_ELO_JSON = paths.DATA / "club_elo.json"


def fetch_club_elo(force: bool = False) -> dict[str, float]:
    """clubelo.com club Elo ratings (free, no key) → {club_name: elo}. Cached."""
    import csv
    import io
    import json as _json
    from datetime import date as _date

    if CLUB_ELO_JSON.exists() and not force:
        return _json.loads(CLUB_ELO_JSON.read_text())
    r = requests.get(CLUBELO_URL.format(date=_date.today().isoformat()), headers=UA, timeout=30)
    r.raise_for_status()
    elo = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            elo[row["Club"]] = round(float(row["Elo"]), 1)
        except (ValueError, KeyError):
            continue
    _write_json_atomic(CLUB_ELO_JSON, _json.dumps(elo, ensure_ascii=False))
    return elo


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORTS = ["soccer_fifa_world_cup", "soccer_fifa_world_cup_2026"]


def _devig3(odds: list[float]) -> list[float]:
    inv = [1.0 / o for o in odds if o and o > 1]
    s = sum(inv)
    return [x / s for x in inv] if len(inv) == 3 and s else []


def fetch_oddsapi_matches() -> dict[tuple, list[float]]:
    """The Odds API (paid key) → per-match 1X2 consensus across ALL bookmakers.

    De-vigs each bookmaker's H/D/A, then averages across books (Pinnacle, Bet365, etc.).
    Returns {(home, away): [pH, pD, pA]}. Empty if no ODDS_API_KEY — activates when you add
    a key. This is the real multi-sportsbook consensus path."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        return {}
    out = {}
    for sport in ODDS_API_SPORTS:
        try:
            r = requests.get(f"{ODDS_API_BASE}/sports/{sport}/odds", params={
                "apiKey": key, "regions": "eu,uk,us", "markets": "h2h",
                "oddsFormat": "decimal"}, headers=UA, timeout=30)
            if r.status_code != 200:
                continue
            for ev in r.json():
                home, away = ev.get("home_team"), ev.get("away_team")
                books = []
                for bk in ev.get("bookmakers", []):
                    for mk in bk.get("markets", []):
                        if mk.get("key") != "h2h":
                            continue
                        o = {x["name"]: x["price"] for x in mk.get("outcomes", [])}
                        trio = [o.get(home), o.get("Draw"), o.get(away)]
                        if all(trio):
                            dv = _devig3(trio)
                            if dv:
                                books.append(dv)
                if books and home and away:
                    avg = [sum(b[i] for b in books) / len(books) for i in range(3)]
                    out[(_canon(home), _canon(away))] = avg
        except requests.RequestException:
            continue
    return out


def fetch_oddsapi_title(team_filter: set[str] | None = None) -> dict[str, float]:
    """The Odds API outrights → {team: de-vigged title prob} (multi-book). Empty w/o key."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        return {}
    for sport in ODDS_API_SPORTS:
        try:
            r = requests.get(f"{ODDS_API_BASE}/sports/{sport}/odds", params={
                "apiKey": key, "regions": "eu,uk,us", "markets": "outrights",
                "oddsFormat": "decimal"}, headers=UA, timeout=30)
            if r.status_code != 200:
                continue
            raw = {}
            for ev in r.json():
                for bk in ev.get("bookmakers", []):
                    for mk in bk.get("markets", []):
                        for o in mk.get("outcomes", []):
                            t = _canon(o["name"])
                            if o.get("price", 0) > 1:
                                raw.setdefault(t, []).append(1.0 / o["price"])
            if raw:
                imp = {t: sum(v) / len(v) for t, v in raw.items()}
                if team_filter:
                    imp = {t: p for t, p in imp.items() if t in team_filter}
                s = sum(imp.values())
                return {t: round(p / s, 5) for t, p in imp.items()} if s else {}
        except requests.RequestException:
            continue
    return {}


def _fd_headers() -> dict:
    return {"X-Auth-Token": os.environ.get("FOOTBALL_DATA_API_KEY", ""), **UA}


FD_MATCHES_JSON = paths.DATA / "fd_matches.json"
# football-data.org nation labels -> our dataset names (where they differ).
def fd_canon(name: str | None) -> str | None:
    if not name:
        return None
    return _canon(name)


def fetch_fd_matches(force: bool = False) -> list[dict[str, Any]]:
    """football-data.org WC matches (all 104, every stage): id, kickoff, status, venue,
    score, teams (knockout teams null until drawn). Cached to disk (free tier = 10 req/min);
    `review` forces a refresh for live scores. Returns cached/[] on error or no key."""
    import json as _json
    if not os.environ.get("FOOTBALL_DATA_API_KEY"):
        return _json.loads(FD_MATCHES_JSON.read_text()) if FD_MATCHES_JSON.exists() else []
    if not force and FD_MATCHES_JSON.exists():
        return _json.loads(FD_MATCHES_JSON.read_text())
    try:
        r = requests.get(f"{paths.FOOTBALL_DATA_BASE}/competitions/WC/matches",
                         headers=_fd_headers(), timeout=30)
        r.raise_for_status()
        ms = r.json().get("matches", [])
        if ms:
            _write_json_atomic(FD_MATCHES_JSON, _json.dumps(ms, ensure_ascii=False))
        return ms
    except requests.RequestException:
        return _json.loads(FD_MATCHES_JSON.read_text()) if FD_MATCHES_JSON.exists() else []


def fetch_fd_match(match_id: int) -> dict[str, Any]:
    """Single match detail — includes lineup + referees once published (match day)."""
    try:
        r = requests.get(f"{paths.FOOTBALL_DATA_BASE}/matches/{match_id}",
                         headers=_fd_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {}


def fd_lineup_absences(match_id: int, squads_team: list[dict]) -> list[str]:
    """Players in the squad NOT in today's confirmed XI (i.e. benched/out) — used to
    down-weight absent key scorers during the tournament. Empty until lineups publish."""
    m = fetch_fd_match(match_id)
    info = m.get("match", m)
    xi = set()
    for side in ("homeTeam", "awayTeam"):
        for p in (info.get(side, {}).get("lineup") or []):
            xi.add(_n(p.get("name", "")))
    if not xi:
        return []
    return sorted({p["name"] for p in squads_team if _n(p["name"]) not in xi})


def _n(s: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch_match_markets() -> dict[tuple, list[float]]:
    """Per-match 1X2 market consensus (de-vigged) keyed by (home, away).

    Combines Polymarket + Kalshi when per-match markets exist. These open close to
    kickoff, so this returns {} pre-tournament — wired so the ensemble activates
    automatically once books/markets list the matches. Defensive: never raises.
    """
    out: dict[tuple, list[float]] = {}
    sources: dict[tuple, int] = {}

    def _add(key, probs):
        if key in out:  # average across markets/books that quote this match
            n = sources[key]
            out[key] = [(out[key][i] * n + probs[i]) / (n + 1) for i in range(3)]
            sources[key] = n + 1
        else:
            out[key] = probs
            sources[key] = 1

    # The Odds API: 50+ bookmakers (Pinnacle/Bet365/...) — multi-book consensus (paid key)
    for k, p in fetch_oddsapi_matches().items():
        _add(k, p)
    # Polymarket: look for 90-minute result markets with H/D/A outcomes
    try:
        r = requests.get(f"{paths.POLYMARKET_GAMMA}/public-search",
                         params={"q": "FIFA World Cup 2026 result"}, headers=UA, timeout=20)
        for ev in r.json().get("events", []):
            title = ev.get("title", "")
            mk = ev.get("markets", [])
            # best-effort: a 3-outcome market (home/draw/away). Format TBD until live.
            for m in mk:
                import json as _json
                try:
                    outs = _json.loads(m.get("outcomes", "[]"))
                    prices = [float(x) for x in _json.loads(m.get("outcomePrices", "[]"))]
                except (ValueError, TypeError):
                    continue
                if len(outs) == 3 and len(prices) == 3 and " vs" in title.lower():
                    teams = title.split(":")[-1].split(" vs")
                    if len(teams) >= 2:
                        h_raw = teams[0].strip()
                        a_raw = teams[1].split("–")[0].strip()
                        h, a = _canon(h_raw), _canon(a_raw)
                        # map prices BY OUTCOME LABEL — never trust positional order.
                        # If the three labels can't be unambiguously matched to
                        # home/draw/away, SKIP: a silently missing market degrades to
                        # model-only; a silently flipped one corrupts the ensemble.
                        lab = [str(o).strip().lower() for o in outs]
                        di = next((i for i, l in enumerate(lab)
                                   if "draw" in l or l in ("tie", "x")), None)
                        def _idx(raw, can):
                            hits = [i for i, l in enumerate(lab)
                                    if l == raw.lower() or _canon(str(outs[i])) == can]
                            return hits[0] if len(hits) == 1 else None
                        hi_, ai_ = _idx(h_raw, h), _idx(a_raw, a)
                        if di is None or hi_ is None or ai_ is None \
                                or len({di, hi_, ai_}) != 3:
                            continue
                        s = sum(prices) or 1
                        _add((h, a), [prices[hi_] / s, prices[di] / s, prices[ai_] / s])
    except (requests.RequestException, ValueError):
        pass
    return out


KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_kalshi_title(team_filter: set[str] | None = None) -> dict[str, float]:
    """Kalshi 'World Cup winner' market → {team: de-vigged prob}. A second market
    signal to blend with Polymarket. Returns {} if Kalshi hasn't listed WC yet."""
    import re

    try:
        evs = requests.get(f"{KALSHI_BASE}/events", params={"limit": 200, "status": "open"},
                           headers=UA, timeout=25).json().get("events", [])
    except (requests.RequestException, ValueError):
        return {}
    wc = [e for e in evs if "world cup" in (e.get("title", "")).lower()
          and "winner" in (e.get("title", "")).lower()]
    raw = {}
    for e in wc:
        try:
            mk = requests.get(f"{KALSHI_BASE}/markets",
                              params={"event_ticker": e.get("event_ticker"), "limit": 100},
                              headers=UA, timeout=25).json().get("markets", [])
        except (requests.RequestException, ValueError):
            continue
        for m in mk:
            # yes_bid/yes_ask in cents → mid price as implied prob
            yb, ya = m.get("yes_bid"), m.get("yes_ask")
            sub = m.get("yes_sub_title") or m.get("subtitle") or ""
            team = _canon(re.sub(r"\s*(to win|winner).*$", "", sub, flags=re.I).strip())
            if yb is not None and ya is not None and team:
                if team_filter and team not in team_filter:
                    continue
                raw[team] = (yb + ya) / 200.0
    s = sum(raw.values())
    return {t: round(p / s, 5) for t, p in raw.items()} if s else {}


def cache_json(name: str, obj: Any) -> None:
    (paths.today_cache() / name).write_text(json.dumps(obj, default=str, indent=2))


def fetch_all() -> dict[str, Any]:
    """One-shot refresh used by `cli fetch --all`."""
    res = load_results()
    fx = load_wc2026_fixtures()
    summary = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "historical_rows": int(len(res)),
        "historical_range": [str(res["date"].min().date()), str(res["date"].max().date())],
        "wc2026_fixtures": int(len(fx)),
    }
    cache_json("fetch_summary.json", summary)
    return summary
