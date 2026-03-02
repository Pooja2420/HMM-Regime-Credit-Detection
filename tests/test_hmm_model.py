"""Tests for the CreditHMM wrapper."""
import numpy as np
import pytest
import tempfile
import os

from src.data.synthetic_generator import generate_credit_data
from src.data.feature_builder import build_features, standardise_features, get_observation_matrix
from src.models.hmm_model import CreditHMM, REGIME_NAMES


@pytest.fixture(scope="module")
def fitted_hmm():
    df, _ = generate_credit_data(n_days=1500, seed=10)
    feat = build_features(df)
    feat_s, _, _ = standardise_features(feat)
    X = get_observation_matrix(feat_s)
    hmm = CreditHMM(n_components=3, n_iter=50, n_init=2, random_state=10)
    hmm.fit(X)
    return hmm, X


def test_fit_returns_self(fitted_hmm):
    hmm, X = fitted_hmm
    assert isinstance(hmm, CreditHMM)
    assert hmm.model is not None


def test_predict_output_shape(fitted_hmm):
    hmm, X = fitted_hmm
    states = hmm.predict(X)
    assert states.shape == (len(X),)


def test_predict_valid_states(fitted_hmm):
    hmm, X = fitted_hmm
    states = hmm.predict(X)
    assert set(states).issubset({0, 1, 2})


def test_predict_proba_shape(fitted_hmm):
    hmm, X = fitted_hmm
    probs = hmm.predict_proba(X)
    assert probs.shape == (len(X), 3)


def test_predict_proba_sums_to_one(fitted_hmm):
    hmm, X = fitted_hmm
    probs = hmm.predict_proba(X)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_predict_proba_non_negative(fitted_hmm):
    hmm, X = fitted_hmm
    probs = hmm.predict_proba(X)
    assert (probs >= 0).all()


def test_score_is_finite(fitted_hmm):
    hmm, X = fitted_hmm
    ll = hmm.score(X)
    assert np.isfinite(ll)
    assert ll < 0  # log-prob must be negative


def test_bic_aic_finite(fitted_hmm):
    hmm, X = fitted_hmm
    assert np.isfinite(hmm.bic(X))
    assert np.isfinite(hmm.aic(X))


def test_transition_matrix_row_stochastic(fitted_hmm):
    hmm, _ = fitted_hmm
    A = hmm.get_transition_matrix()
    row_sums = A.sum(axis=1).values
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_transition_matrix_index(fitted_hmm):
    hmm, _ = fitted_hmm
    A = hmm.get_transition_matrix()
    assert list(A.index) == REGIME_NAMES
    assert list(A.columns) == REGIME_NAMES


def test_emission_params_shapes(fitted_hmm):
    hmm, X = fitted_hmm
    params = hmm.get_emission_params()
    assert params["means"].shape == (3, X.shape[1])
    assert params["covars"].shape == (3, X.shape[1], X.shape[1])


def test_save_load_roundtrip(fitted_hmm):
    hmm, X = fitted_hmm
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test_hmm.joblib")
        hmm.save(path)
        hmm2 = CreditHMM.load(path)
        np.testing.assert_allclose(hmm.score(X), hmm2.score(X), rtol=1e-6)


def test_two_state_model():
    df, _ = generate_credit_data(n_days=800, seed=11)
    feat = build_features(df)
    feat_s, _, _ = standardise_features(feat)
    X = get_observation_matrix(feat_s)
    hmm2 = CreditHMM(n_components=2, n_iter=30, n_init=1, random_state=11)
    hmm2.fit(X)
    states = hmm2.predict(X)
    assert set(states).issubset({0, 1})
