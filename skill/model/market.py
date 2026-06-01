"""Market prior: de-vig bookmaker odds and blend with the statistical model.

The de-vig (overround removal) and Kelly math is implemented here so the repo has
no hard dependency on an external betting package.
"""
from __future__ import annotations

import numpy as np


def implied_from_decimal(odds: list[float]) -> np.ndarray:
    return np.array([1.0 / o for o in odds], dtype=float)


def devig_proportional(odds: list[float]) -> np.ndarray:
    """Simplest de-vig: normalise inverse-odds so they sum to 1."""
    p = implied_from_decimal(odds)
    return p / p.sum()


def devig_power(odds: list[float], tol: float = 1e-9) -> np.ndarray:
    """Power method: find k so sum(p_i**k) = 1. Reduces favourite-longshot bias
    better than proportional. Falls back to proportional if it fails."""
    raw = implied_from_decimal(odds)
    lo, hi = 0.5, 2.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        s = np.sum(raw**k)
        if abs(s - 1) < tol:
            break
        if s > 1:
            lo = k
        else:
            hi = k
    out = raw**k
    out = out / out.sum()
    return out if np.all(np.isfinite(out)) else devig_proportional(odds)


def consensus_market(books: list[list[float]], method: str = "power") -> np.ndarray | None:
    """Average de-vigged probabilities across bookmakers (1X2 order H,D,A)."""
    if not books:
        return None
    fn = devig_power if method == "power" else devig_proportional
    probs = np.array([fn(b) for b in books if b and len(b) == 3])
    if probs.size == 0:
        return None
    return probs.mean(axis=0)


def polymarket_prob(yes_price: float) -> float:
    """Polymarket YES price (0..1) is already a probability."""
    return float(np.clip(yes_price, 0.0, 1.0))


def blend(p_market: np.ndarray | None, p_model: np.ndarray, w: float) -> np.ndarray:
    """P_final = w*market + (1-w)*model. If no market, fall back to model."""
    p_model = np.asarray(p_model, dtype=float)
    p_model = p_model / p_model.sum()
    if p_market is None:
        return p_model
    p_market = np.asarray(p_market, dtype=float)
    p_market = p_market / p_market.sum()
    out = w * p_market + (1.0 - w) * p_model
    return out / out.sum()


def edge(p_final: np.ndarray, p_market: np.ndarray | None) -> np.ndarray | None:
    if p_market is None:
        return None
    return np.asarray(p_final) - np.asarray(p_market)


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Fraction of bankroll for a single outcome (0 if no edge)."""
    b = decimal_odds - 1.0
    f = (p * b - (1.0 - p)) / b if b > 0 else 0.0
    return max(0.0, float(f))
