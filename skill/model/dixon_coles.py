"""Dixon-Coles bivariate Poisson with exponential time decay.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market." We fit team attack/defence
strengths + home advantage by weighted MLE, with the DC low-score correction
(rho) and an exponential time-decay weight (xi). A small ridge penalty on the
strength vectors doubles as an overfitting guard.

The fit only ever sees matches with date <= as_of, so it is look-ahead free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class DCModel:
    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_adv: float
    rho: float
    xi: float
    intercept: float

    def lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        ah = self.attack.get(home, 0.0)
        aa = self.attack.get(away, 0.0)
        dh = self.defence.get(home, 0.0)
        da = self.defence.get(away, 0.0)
        hadv = 0.0 if neutral else self.home_adv
        lam = np.exp(self.intercept + hadv + ah - da)
        mu = np.exp(self.intercept + aa - dh)
        return float(lam), float(mu)


def _tau(h: np.ndarray, a: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """DC low-score correlation correction."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((h == 0) & (a == 0), 1.0 - lam * mu * rho, t)
    t = np.where((h == 0) & (a == 1), 1.0 + lam * rho, t)
    t = np.where((h == 1) & (a == 0), 1.0 + mu * rho, t)
    t = np.where((h == 1) & (a == 1), 1.0 - rho, t)
    return np.clip(t, 1e-9, None)


def fit(
    results: pd.DataFrame,
    as_of: pd.Timestamp,
    xi: float = 0.0010,
    ridge: float = 0.01,
    min_matches: int = 8,
    train_years: float = 8.0,
) -> DCModel:
    """Weighted MLE on matches strictly before `as_of`. `xi` is daily decay.

    Matches older than `train_years` carry negligible decay weight, so we drop
    them — large speedup with no material effect on the fit.
    """
    df = results[results["date"] < as_of].copy()
    if train_years:
        df = df[df["date"] >= as_of - pd.Timedelta(days=int(365.25 * train_years))]
    if df.empty:
        raise ValueError("no training data before as_of")

    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    teams = sorted(counts[counts >= min_matches].index.tolist())
    if len(teams) < 2:
        raise ValueError("not enough teams with sufficient matches")
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)].copy()

    tidx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["home_team"].map(tidx).to_numpy()
    ai = df["away_team"].map(tidx).to_numpy()
    hs = df["home_score"].to_numpy(dtype=int)
    as_ = df["away_score"].to_numpy(dtype=int)
    neutral = df["neutral"].to_numpy(dtype=bool)
    days = (as_of - df["date"]).dt.days.to_numpy(dtype=float)
    w = np.exp(-xi * days)

    # params: [intercept, home_adv, attack(n-1 free), defence(n-1 free), rho]
    # last team's attack/defence pinned to 0 for identifiability.
    def unpack(p):
        intercept = p[0]
        hadv = p[1]
        atk = np.append(p[2 : 2 + (n - 1)], 0.0)
        dfc = np.append(p[2 + (n - 1) : 2 + 2 * (n - 1)], 0.0)
        rho = p[-1]
        return intercept, hadv, atk, dfc, rho

    def negll(p):
        intercept, hadv, atk, dfc, rho = unpack(p)
        hadv_eff = np.where(neutral, 0.0, hadv)
        lam = np.exp(intercept + hadv_eff + atk[hi] - dfc[ai])
        mu = np.exp(intercept + atk[ai] - dfc[hi])
        lam = np.clip(lam, 1e-6, 25)
        mu = np.clip(mu, 1e-6, 25)
        # Poisson log-pmf (drop constant log(k!) — irrelevant to optimum).
        ll_h = hs * np.log(lam) - lam
        ll_a = as_ * np.log(mu) - mu
        ll_tau = np.log(_tau(hs, as_, lam, mu, rho))
        ll = w * (ll_h + ll_a + ll_tau)
        pen = ridge * (np.sum(atk**2) + np.sum(dfc**2))
        return -(ll.sum()) + pen

    x0 = np.zeros(2 + 2 * (n - 1) + 1)
    x0[0] = 0.0  # intercept
    x0[1] = 0.25  # home adv
    x0[-1] = -0.05  # rho
    bounds = [(-2, 2), (-1, 1)] + [(-3, 3)] * (2 * (n - 1)) + [(-0.2, 0.2)]
    res = minimize(negll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 400, "maxfun": 40000})
    intercept, hadv, atk, dfc, rho = unpack(res.x)
    # centre strengths for interpretability
    atk = atk - atk.mean()
    dfc = dfc - dfc.mean()
    return DCModel(
        teams=teams,
        attack={t: float(atk[i]) for t, i in tidx.items()},
        defence={t: float(dfc[i]) for t, i in tidx.items()},
        home_adv=float(hadv),
        rho=float(rho),
        xi=xi,
        intercept=float(intercept),
    )


def scoreline_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """P(home=i, away=j) with DC correction, normalised."""
    from scipy.stats import poisson

    h = poisson.pmf(np.arange(max_goals + 1), lam)
    a = poisson.pmf(np.arange(max_goals + 1), mu)
    m = np.outer(h, a)
    # apply tau to the 2x2 low-score corner
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def match_probs(model: DCModel, home: str, away: str, neutral: bool = True,
                max_goals: int = 10, lam_mult: float = 1.0, mu_mult: float = 1.0) -> dict:
    """Full set of market probabilities for one fixture.

    lam_mult/mu_mult apply situational context adjustments (altitude/rest/travel)
    to each side's expected goals before building the scoreline distribution.
    """
    lam, mu = model.lambdas(home, away, neutral)
    lam, mu = lam * lam_mult, mu * mu_mult
    m = scoreline_matrix(lam, mu, model.rho, max_goals)
    idx = np.arange(max_goals + 1)
    p_home = float(np.tril(m, -1).sum())
    p_draw = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())
    total = idx[:, None] + idx[None, :]
    p_over25 = float(m[total >= 3].sum())
    p_btts = float(m[1:, 1:].sum())
    top = np.dstack(np.unravel_index(np.argsort(m.ravel())[::-1][:5], m.shape))[0]
    return {
        "home": home, "away": away, "neutral": neutral,
        "lambda_home": round(lam, 3), "lambda_away": round(mu, 3),
        "p_home": round(p_home, 4), "p_draw": round(p_draw, 4), "p_away": round(p_away, 4),
        "p_over_2_5": round(p_over25, 4), "p_under_2_5": round(1 - p_over25, 4),
        "p_btts": round(p_btts, 4),
        "top_scorelines": [
            {"score": f"{int(i)}-{int(j)}", "p": round(float(m[i, j]), 4)} for i, j in top
        ],
    }
