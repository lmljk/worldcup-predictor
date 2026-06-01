"""Probabilistic scoring metrics for 1X2 forecasts."""
from __future__ import annotations

import numpy as np


def rps(probs: np.ndarray, outcome: int) -> float:
    """Ranked Probability Score for ordered outcomes (home=0, draw=1, away=2).

    The standard football metric (Constantinou & Fenton 2012). Lower is better.
    """
    p = np.asarray(probs, dtype=float)
    obs = np.zeros_like(p)
    obs[outcome] = 1.0
    cp = np.cumsum(p)
    co = np.cumsum(obs)
    return float(np.sum((cp - co) ** 2) / (len(p) - 1))


def brier(probs: np.ndarray, outcome: int) -> float:
    p = np.asarray(probs, dtype=float)
    obs = np.zeros_like(p)
    obs[outcome] = 1.0
    return float(np.mean((p - obs) ** 2))


def log_loss(probs: np.ndarray, outcome: int, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    return float(-np.log(p[outcome]))


def outcome_index(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-match metric dicts into means."""
    if not rows:
        return {}
    keys = ["rps", "brier", "log_loss"]
    out = {f"mean_{k}": round(float(np.mean([r[k] for r in rows])), 5) for k in keys}
    out["n"] = len(rows)
    hits = np.mean([np.argmax(r["probs"]) == r["outcome"] for r in rows])
    out["top_pick_accuracy"] = round(float(hits), 4)
    return out
