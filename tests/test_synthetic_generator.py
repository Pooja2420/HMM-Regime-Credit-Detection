"""Tests for synthetic data generator."""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import (
    generate_credit_data, DEFAULT_TRANSITION, REGIME_NAMES,
)


def test_output_shape():
    df, states = generate_credit_data(n_days=500, seed=0)
    assert len(df) == 500
    assert len(states) == 500
    assert set(df.columns) == {"cdx_ig", "cdx_hy", "vix", "true_regime"}


def test_output_index_is_business_days():
    df, _ = generate_credit_data(n_days=100, start_date="2020-01-01", seed=1)
    assert isinstance(df.index, pd.DatetimeIndex)
    # All weekdays (0=Mon ... 4=Fri)
    assert all(d.weekday() < 5 for d in df.index)


def test_spread_ranges_non_negative():
    df, _ = generate_credit_data(n_days=1000, seed=2)
    assert (df["cdx_ig"] > 0).all()
    assert (df["cdx_hy"] > 0).all()
    assert (df["vix"] > 0).all()


def test_states_valid():
    _, states = generate_credit_data(n_days=1000, seed=3)
    assert states.min() >= 0
    assert states.max() <= 2
    assert set(states).issubset({0, 1, 2})


def test_all_regimes_present_long_series():
    _, states = generate_credit_data(n_days=3000, seed=4)
    assert len(set(states)) == 3, "All 3 regimes should appear in a long series"


def test_reproducibility():
    df1, s1 = generate_credit_data(n_days=200, seed=42)
    df2, s2 = generate_credit_data(n_days=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)
    np.testing.assert_array_equal(s1, s2)


def test_different_seeds_differ():
    df1, _ = generate_credit_data(n_days=200, seed=1)
    df2, _ = generate_credit_data(n_days=200, seed=2)
    assert not df1["cdx_ig"].equals(df2["cdx_ig"])


def test_transition_matrix_row_stochastic():
    A = DEFAULT_TRANSITION
    row_sums = A.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_tight_lower_spreads_than_stress():
    df, states = generate_credit_data(n_days=3000, seed=5)
    tight_ig = df.loc[df["true_regime"] == 0, "cdx_ig"].mean()
    stress_ig = df.loc[df["true_regime"] == 2, "cdx_ig"].mean()
    assert tight_ig < stress_ig, "Tight should have lower IG spreads than Stress"


def test_stress_higher_vix():
    df, states = generate_credit_data(n_days=3000, seed=6)
    normal_vix = df.loc[df["true_regime"] == 1, "vix"].mean()
    stress_vix = df.loc[df["true_regime"] == 2, "vix"].mean()
    assert stress_vix > normal_vix
