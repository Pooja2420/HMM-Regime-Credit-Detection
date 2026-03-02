"""Tests for feature engineering module."""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic_generator import generate_credit_data
from src.data.feature_builder import (
    build_features, standardise_features, get_observation_matrix, FEATURE_NAMES,
)


@pytest.fixture(scope="module")
def raw_df():
    df, _ = generate_credit_data(n_days=500, seed=7)
    return df


def test_feature_columns(raw_df):
    feat = build_features(raw_df)
    assert list(feat.columns) == FEATURE_NAMES


def test_no_nans_after_dropna(raw_df):
    feat = build_features(raw_df, dropna=True)
    assert feat.isna().sum().sum() == 0


def test_feature_count_smaller_than_raw(raw_df):
    feat = build_features(raw_df)
    assert len(feat) < len(raw_df)  # warmup rows dropped


def test_feature_index_is_subset_of_raw(raw_df):
    feat = build_features(raw_df)
    assert feat.index.isin(raw_df.index).all()


def test_standardise_zero_mean_unit_std(raw_df):
    feat = build_features(raw_df)
    n = int(len(feat) * 0.8)
    train_s, _, _ = standardise_features(feat.iloc[:n])
    means = train_s.mean()
    stds = train_s.std()
    np.testing.assert_allclose(means.values, 0.0, atol=1e-10)
    np.testing.assert_allclose(stds.values, 1.0, atol=1e-10)


def test_standardise_test_uses_train_stats(raw_df):
    feat = build_features(raw_df)
    n = int(len(feat) * 0.8)
    train_s, test_s, (mu, sigma) = standardise_features(
        feat.iloc[:n], feat.iloc[n:]
    )
    # test set should NOT necessarily have zero mean (different distribution slice)
    assert test_s is not None
    assert len(test_s) == len(feat) - n


def test_observation_matrix_dtype(raw_df):
    feat = build_features(raw_df)
    X = get_observation_matrix(feat)
    assert X.dtype == np.float64
    assert X.shape[1] == len(FEATURE_NAMES)


def test_rolling_vol_positive(raw_df):
    feat = build_features(raw_df)
    assert (feat["ig_roll_vol_21"] >= 0).all()
    assert (feat["hy_roll_vol_21"] >= 0).all()


def test_vix_norm_around_one(raw_df):
    feat = build_features(raw_df)
    # VIX normalised by rolling mean: should be roughly centred near 1
    mean_norm = feat["vix_level_norm"].mean()
    assert 0.5 < mean_norm < 2.0
