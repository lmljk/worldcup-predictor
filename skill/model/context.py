"""Situational context adjustments (free, deterministic): altitude, rest, travel.

These are small multiplicative nudges to each side's expected goals (lambda), computed
from the fixture schedule + a static venue table — no paid data. Like the talent prior,
they are transparent priors (WC samples are too small to walk-forward validate), so the
magnitudes are deliberately modest.

Weather/heat is intentionally NOT applied pre-tournament: Open-Meteo only forecasts ~16
days out, so it only becomes usable during the event (wired into the review loop later).
"""
from __future__ import annotations

import math

# 2026 host venues by the city string used in the fixture data.
VENUES = {
    "Arlington": {"alt": 150, "lat": 32.747, "lon": -97.093},
    "Atlanta": {"alt": 320, "lat": 33.755, "lon": -84.401},
    "East Rutherford": {"alt": 5, "lat": 40.813, "lon": -74.074},
    "Foxborough": {"alt": 30, "lat": 42.091, "lon": -71.264},
    "Guadalupe": {"alt": 500, "lat": 25.669, "lon": -100.244},      # Monterrey
    "Houston": {"alt": 15, "lat": 29.685, "lon": -95.411},
    "Inglewood": {"alt": 30, "lat": 33.953, "lon": -118.339},       # LA
    "Kansas City": {"alt": 270, "lat": 39.049, "lon": -94.484},
    "Mexico City": {"alt": 2240, "lat": 19.303, "lon": -99.150},    # Estadio Azteca — high
    "Miami Gardens": {"alt": 3, "lat": 25.958, "lon": -80.239},
    "Philadelphia": {"alt": 10, "lat": 39.901, "lon": -75.168},
    "Santa Clara": {"alt": 5, "lat": 37.403, "lon": -121.970},
    "Seattle": {"alt": 40, "lat": 47.595, "lon": -122.332},
    "Toronto": {"alt": 76, "lat": 43.633, "lon": -79.418},
    "Vancouver": {"alt": 5, "lat": 49.277, "lon": -123.112},
    "Zapopan": {"alt": 1566, "lat": 20.681, "lon": -103.463},       # Guadalajara — moderate
}
# teams acclimatised to high altitude (no altitude penalty)
ALTITUDE_ACCLIM = {"Mexico", "Ecuador"}


def _haversine(a, b) -> float:
    R = 6371.0
    dlat = math.radians(b["lat"] - a["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def _altitude_penalty(alt: float) -> float:
    if alt < 1300:
        return 0.0
    return min(0.10, (alt - 1000) / 1000.0 * 0.05)  # 2240m → ~6%


def _travel_penalty(dist: float) -> float:
    return min(0.04, max(0.0, (dist - 1500) / 1000.0 * 0.012))


def compute(fixtures) -> dict[str, dict]:
    """Per-fixture {fixture_id: {home_mult, away_mult, notes[]}} over the schedule."""
    out = {}
    last: dict[str, tuple] = {}  # team -> (date, city)
    for r in fixtures.sort_values("date").itertuples(index=False):
        fid = r.fixture_id
        home, away, city = r.home_team, r.away_team, r.city
        v = VENUES.get(city)
        hm = am = 1.0
        notes = []

        # altitude
        if v:
            pen = _altitude_penalty(v["alt"])
            if pen > 0:
                if home not in ALTITUDE_ACCLIM:
                    hm *= (1 - pen)
                if away not in ALTITUDE_ACCLIM:
                    am *= (1 - pen)
                if pen >= 0.03:
                    notes.append(f"altitude {int(v['alt'])}m")

        # rest-day differential
        rd = {}
        for tm in (home, away):
            prev = last.get(tm)
            rd[tm] = (r.date - prev[0]).days if prev else 10
        diff = rd[home] - rd[away]
        if abs(diff) >= 2:
            pen = min(0.06, abs(diff) * 0.015)
            if diff > 0:
                am *= (1 - pen)
                notes.append(f"{away} short rest")
            else:
                hm *= (1 - pen)
                notes.append(f"{home} short rest")

        # travel since previous match
        if v:
            for tm, mult_is_home in ((home, True), (away, False)):
                prev = last.get(tm)
                if prev and prev[1] in VENUES:
                    dist = _haversine(VENUES[prev[1]], v)
                    pen = _travel_penalty(dist)
                    if pen > 0:
                        if mult_is_home:
                            hm *= (1 - pen)
                        else:
                            am *= (1 - pen)

        out[fid] = {"home_mult": round(hm, 4), "away_mult": round(am, 4), "notes": notes}
        last[home] = (r.date, city)
        last[away] = (r.date, city)
    return out
