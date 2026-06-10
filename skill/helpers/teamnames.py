"""One authoritative team-name canonicalisation table.

Canonical spelling = the martj42 dataset names (the modelling backbone). Every
external source (Polymarket, Kalshi, The Odds API, Wikipedia squads,
football-data.org, EA FC25) must go through `canon()` — previously each source
kept its own partial alias dict, so a team could silently fail to join across
sources whenever one dict was missing a variant (e.g. "Cabo Verde" was only
known to the Polymarket table).
"""
from __future__ import annotations

ALIASES = {
    # United States
    "USA": "United States", "U.S.": "United States", "US": "United States",
    "United States of America": "United States",
    # Czech Republic
    "Czechia": "Czech Republic",
    # South Korea
    "Korea Republic": "South Korea", "Republic of Korea": "South Korea",
    "Korea, South": "South Korea",
    # Iran
    "IR Iran": "Iran", "Iran IR": "Iran",
    # Turkey
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    # Ivory Coast
    "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Cote dIvoire": "Ivory Coast",
    # Cape Verde
    "Cape Verde Islands": "Cape Verde", "Cabo Verde": "Cape Verde",
    # DR Congo
    "Congo DR": "DR Congo", "Congo, DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    # Bosnia and Herzegovina
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    # Curaçao (martj42 keeps the accent)
    "Curacao": "Curaçao",
}


def canon(name: str | None) -> str | None:
    """Map any known variant to the canonical (martj42) team name; pass through
    unknowns unchanged, None stays None."""
    if not name:
        return name
    return ALIASES.get(str(name).strip(), str(name).strip())
