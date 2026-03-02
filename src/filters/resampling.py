"""
Resampling strategies for Sequential Monte Carlo.

Both systematic and stratified resampling have O(N) complexity
and produce lower-variance estimates than multinomial resampling.
"""

import numpy as np


def systematic_resample(
    weights: np.ndarray, N: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Systematic resampling (Kitagawa 1996).

    Draws a single uniform random number and selects N indices at evenly
    spaced intervals along the cumulative weight distribution.  Guarantees
    that each interval [k/N, (k+1)/N) contributes exactly one sample.

    Parameters
    ----------
    weights : np.ndarray, shape (N,), normalised to sum to 1.
    N : int   Number of particles to resample.
    rng : np.random.Generator

    Returns
    -------
    indices : np.ndarray, shape (N,)  Selected particle indices.
    """
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # numerical safety

    start = rng.uniform(0, 1.0 / N)
    positions = start + np.arange(N) / N

    indices = np.searchsorted(cumsum, positions)
    return indices


def stratified_resample(
    weights: np.ndarray, N: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Stratified resampling.

    Divides [0, 1] into N equal strata and draws one sample uniformly
    from each stratum, then maps to particle index via inverse CDF.

    Parameters
    ----------
    weights : np.ndarray, shape (N,), normalised to sum to 1.
    N : int   Number of particles to resample.
    rng : np.random.Generator

    Returns
    -------
    indices : np.ndarray, shape (N,)  Selected particle indices.
    """
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0

    strata_starts = np.arange(N) / N
    uniform_within = rng.uniform(0, 1.0 / N, size=N)
    positions = strata_starts + uniform_within

    indices = np.searchsorted(cumsum, positions)
    return indices


def multinomial_resample(
    weights: np.ndarray, N: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Standard multinomial resampling (high variance baseline).
    """
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    positions = np.sort(rng.uniform(0, 1, size=N))
    return np.searchsorted(cumsum, positions)
