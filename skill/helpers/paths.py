"""Path constants and shared config for the worldcup skill."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HISTORICAL = DATA / "historical"
CACHE = ROOT / "data_cache"
REPORTS = ROOT / "reports"
BACKTESTS = REPORTS / "backtests"
SITE = ROOT / "site"

for _d in (HISTORICAL, CACHE, REPORTS, BACKTESTS, SITE):
    _d.mkdir(parents=True, exist_ok=True)

# Public, no-auth data sources.
MARTJ42_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

# Optional keys (read from .env if present).
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

WC2026_TOURNAMENT = "FIFA World Cup"
WC2026_START = date(2026, 6, 1)

HISTORICAL_RESULTS_CSV = HISTORICAL / "results.csv"
VENUES_JSON = DATA / "venues_wc2026.json"
FIXTURES_JSON = DATA / "fixtures_wc2026.json"


def today_cache() -> Path:
    d = CACHE / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(d: str | None = None) -> Path:
    d = d or date.today().isoformat()
    p = REPORTS / d
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dotenv() -> None:
    """Lightweight .env loader (no hard dependency on python-dotenv)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
