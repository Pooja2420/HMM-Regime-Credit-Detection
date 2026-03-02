"""
Synthetic regime-switching credit spread generator.

Produces realistic CDX IG, CDX HY, and VIX time series driven by a latent
3-state Markov chain: Tight (0), Normal (1), Stress (2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Regime parameters — calibrated to approximate historical CDX dynamics
# ---------------------------------------------------------------------------

@dataclass
class RegimeParams:
    """Emission parameters for one regime."""
    name: str
    # CDX IG OAS (bps)
    ig_mean: float
    ig_vol: float          # daily vol in bps
    ig_autocorr: float     # AR(1) coefficient for mean-reversion
    # CDX HY OAS (bps)
    hy_mean: float
    hy_vol: float
    hy_autocorr: float
    # VIX level
    vix_mean: float
    vix_vol: float
    vix_autocorr: float
    # Cross-asset correlation matrix [IG_chg, HY_chg, VIX_chg]
    corr: np.ndarray = field(default_factory=lambda: np.eye(3))


REGIME_PARAMS = [
    RegimeParams(
        name="Tight",
        ig_mean=60.0,   ig_vol=1.5,  ig_autocorr=0.92,
        hy_mean=270.0,  hy_vol=6.0,  hy_autocorr=0.90,
        vix_mean=13.0,  vix_vol=0.8, vix_autocorr=0.93,
        corr=np.array([
            [1.00,  0.80,  0.30],
            [0.80,  1.00,  0.35],
            [0.30,  0.35,  1.00],
        ]),
    ),
    RegimeParams(
        name="Normal",
        ig_mean=110.0,  ig_vol=3.0,  ig_autocorr=0.88,
        hy_mean=380.0,  hy_vol=12.0, hy_autocorr=0.86,
        vix_mean=18.0,  vix_vol=1.5, vix_autocorr=0.90,
        corr=np.array([
            [1.00,  0.85,  0.50],
            [0.85,  1.00,  0.55],
            [0.50,  0.55,  1.00],
        ]),
    ),
    RegimeParams(
        name="Stress",
        ig_mean=210.0,  ig_vol=8.0,  ig_autocorr=0.82,
        hy_mean=650.0,  hy_vol=30.0, hy_autocorr=0.80,
        vix_mean=32.0,  vix_vol=4.0, vix_autocorr=0.85,
        corr=np.array([
            [1.00,  0.92,  0.75],
            [0.92,  1.00,  0.78],
            [0.75,  0.78,  1.00],
        ]),
    ),
]

# Transition matrix: stress is persistent, tight exits quickly
DEFAULT_TRANSITION = np.array([
    [0.92, 0.07, 0.01],   # Tight -> Tight / Normal / Stress
    [0.04, 0.93, 0.03],   # Normal -> ...
    [0.02, 0.10, 0.88],   # Stress -> ...
])

REGIME_NAMES = ["Tight", "Normal", "Stress"]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_credit_data(
    n_days: int = 3500,
    start_date: str = "2010-01-04",
    transition_matrix: Optional[np.ndarray] = None,
    regime_params: Optional[list] = None,
    initial_state: int = 1,
    seed: Optional[int] = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic daily credit spread data with latent regime.

    Parameters
    ----------
    n_days : int
        Number of business days to simulate.
    start_date : str
        Start date string (ISO format).
    transition_matrix : np.ndarray, optional
        3x3 row-stochastic transition matrix.
    regime_params : list of RegimeParams, optional
        Override default emission parameters.
    initial_state : int
        Starting regime (0=Tight, 1=Normal, 2=Stress).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Columns: date, cdx_ig, cdx_hy, vix, true_regime
    true_regimes : np.ndarray
        Integer array of true latent state per day.
    """
    rng = np.random.default_rng(seed)
    A = transition_matrix if transition_matrix is not None else DEFAULT_TRANSITION
    params = regime_params if regime_params is not None else REGIME_PARAMS

    # --- Simulate Markov chain ---
    states = np.empty(n_days, dtype=int)
    states[0] = initial_state
    for t in range(1, n_days):
        states[t] = rng.choice(3, p=A[states[t - 1]])

    # --- Simulate observations via correlated AR(1) processes ---
    ig = np.zeros(n_days)
    hy = np.zeros(n_days)
    vix_arr = np.zeros(n_days)

    p = params[states[0]]
    ig[0] = p.ig_mean + rng.normal(0, p.ig_vol)
    hy[0] = p.hy_mean + rng.normal(0, p.hy_vol)
    vix_arr[0] = p.vix_mean + rng.normal(0, p.vix_vol)

    for t in range(1, n_days):
        p = params[states[t]]
        prev_p = params[states[t - 1]]

        # Correlated innovations via Cholesky decomposition
        vols = np.array([p.ig_vol, p.hy_vol, p.vix_vol])
        L = np.linalg.cholesky(p.corr)
        z = L @ rng.standard_normal(3)

        # AR(1) mean-reversion around regime mean
        ig[t] = (p.ig_autocorr * ig[t - 1]
                 + (1 - p.ig_autocorr) * p.ig_mean
                 + vols[0] * z[0])
        hy[t] = (p.hy_autocorr * hy[t - 1]
                 + (1 - p.hy_autocorr) * p.hy_mean
                 + vols[1] * z[1])
        vix_arr[t] = (p.vix_autocorr * vix_arr[t - 1]
                      + (1 - p.vix_autocorr) * p.vix_mean
                      + vols[2] * z[2])

    # Enforce non-negativity
    ig = np.clip(ig, 20, 600)
    hy = np.clip(hy, 100, 2000)
    vix_arr = np.clip(vix_arr, 8, 80)

    # --- Build DataFrame ---
    dates = pd.bdate_range(start=start_date, periods=n_days)
    df = pd.DataFrame({
        "date": dates,
        "cdx_ig": np.round(ig, 2),
        "cdx_hy": np.round(hy, 2),
        "vix": np.round(vix_arr, 2),
        "true_regime": states,
    })
    df.set_index("date", inplace=True)

    return df, states


def get_regime_colors() -> dict:
    """Return consistent color mapping for regimes."""
    return {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}


def get_regime_labels() -> dict:
    return {0: "Tight", 1: "Normal", 2: "Stress"}
