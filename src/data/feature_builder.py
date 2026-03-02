"""
Feature engineering for HMM observation vectors.

Transforms raw spread / VIX levels into the multi-dimensional observation
space used by the HMM emission model and particle filter likelihood.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple


FEATURE_NAMES = [
    "ig_return",          # Daily log-return of CDX IG spread
    "hy_return",          # Daily log-return of CDX HY spread
    "ig_roll_vol_21",     # 21-day rolling stddev of IG returns
    "hy_roll_vol_21",     # 21-day rolling stddev of HY returns
    "hy_ig_spread",       # HY - IG spread differential (credit risk premium)
    "vix_level_norm",     # VIX normalised by its 252-day rolling mean
    "ig_momentum_5",      # 5-day IG spread change (momentum)
    "hy_momentum_5",      # 5-day HY spread change
    "ig_autocorr_10",     # 10-day rolling autocorrelation of IG returns
]


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Compute HMM observation features from raw spread data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: cdx_ig, cdx_hy, vix.
        Index should be a DatetimeIndex.
    dropna : bool
        Drop rows with NaN (warm-up period for rolling features).

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned with input index, NaN rows optionally dropped.
    """
    feat = pd.DataFrame(index=df.index)

    # --- Returns ---
    feat["ig_return"] = np.log(df["cdx_ig"] / df["cdx_ig"].shift(1))
    feat["hy_return"] = np.log(df["cdx_hy"] / df["cdx_hy"].shift(1))

    # --- Rolling volatility (annualised sqrt-252 scaling removed; raw daily) ---
    feat["ig_roll_vol_21"] = feat["ig_return"].rolling(21).std()
    feat["hy_roll_vol_21"] = feat["hy_return"].rolling(21).std()

    # --- Spread differential ---
    feat["hy_ig_spread"] = df["cdx_hy"] - df["cdx_ig"]

    # --- VIX normalised ---
    vix_ma = df["vix"].rolling(252, min_periods=63).mean()
    feat["vix_level_norm"] = df["vix"] / vix_ma

    # --- Momentum ---
    feat["ig_momentum_5"] = df["cdx_ig"].diff(5)
    feat["hy_momentum_5"] = df["cdx_hy"].diff(5)

    # --- Rolling autocorrelation ---
    feat["ig_autocorr_10"] = (
        feat["ig_return"]
        .rolling(30)
        .apply(lambda x: x.autocorr(lag=1) if len(x) > 2 else np.nan, raw=False)
    )

    if dropna:
        feat.dropna(inplace=True)

    return feat[FEATURE_NAMES]


def standardise_features(
    train_feat: pd.DataFrame,
    test_feat: Optional[pd.DataFrame] = None,
) -> tuple:
    """
    Z-score standardise features using training-set statistics.

    Returns
    -------
    train_scaled, test_scaled (or None), (mean, std) scaler tuple
    """
    mu = train_feat.mean()
    sigma = train_feat.std().replace(0, 1)

    train_scaled = (train_feat - mu) / sigma

    if test_feat is not None:
        test_scaled = (test_feat - mu) / sigma
        return train_scaled, test_scaled, (mu, sigma)

    return train_scaled, None, (mu, sigma)


def get_observation_matrix(feat_df: pd.DataFrame) -> np.ndarray:
    """Return feature DataFrame as float64 numpy array for hmmlearn."""
    return feat_df.to_numpy(dtype=np.float64)
