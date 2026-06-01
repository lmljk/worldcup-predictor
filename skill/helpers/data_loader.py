"""Data acquisition: historical results, WC2026 fixtures, weather, Polymarket.

All sources here are public and need no auth. Everything caches to disk so backtests
are reproducible and we never hammer an API.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from . import paths

paths.load_dotenv()

UA = {"User-Agent": "worldcup-predictor/0.1 (+https://github.com/rrclaw)"}


def fetch_historical(force: bool = False) -> pd.DataFrame:
    """martj42 international results (1872-now) + WC2026 fixtures in one CSV."""
    csv = paths.HISTORICAL_RESULTS_CSV
    if force or not csv.exists():
        r = requests.get(paths.MARTJ42_RESULTS_URL, headers=UA, timeout=60)
        r.raise_for_status()
        csv.write_bytes(r.content)
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


# Polymarket uses slightly different country labels than the martj42 dataset.
PM_NAME_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "USA": "United States",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
}


def _canon(team: str) -> str:
    return PM_NAME_ALIASES.get(team, team)


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
