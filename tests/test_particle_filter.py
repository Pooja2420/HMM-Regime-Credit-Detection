"""Tests for the SMC particle filter and resampling strategies."""
import numpy as np
import pytest

from src.data.synthetic_generator import generate_credit_data, DEFAULT_TRANSITION
from src.data.feature_builder import build_features, standardise_features, get_observation_matrix
from src.models.hmm_model import CreditHMM
from src.filters.particle_filter import ParticleFilter
from src.filters.resampling import systematic_resample, stratified_resample, multinomial_resample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pf_setup():
    df, _ = generate_credit_data(n_days=600, seed=20)
    feat = build_features(df)
    feat_s, _, _ = standardise_features(feat)
    X = get_observation_matrix(feat_s)

    hmm = CreditHMM(n_components=3, n_iter=50, n_init=2, random_state=20)
    hmm.fit(X)

    params = hmm.get_emission_params()
    A = hmm.get_transition_matrix().to_numpy()

    pf = ParticleFilter(
        n_particles=500,
        transition_matrix=A,
        means=params["means"],
        covars=params["covars"],
        seed=20,
    )
    return pf, X


# ---------------------------------------------------------------------------
# Particle filter tests
# ---------------------------------------------------------------------------

def test_initialise_uniform(pf_setup):
    pf, X = pf_setup
    pf.initialise()
    assert pf.particles.shape == (500,)
    assert pf.weights.shape == (500,)
    np.testing.assert_allclose(pf.weights.sum(), 1.0, atol=1e-10)


def test_initialise_with_prior(pf_setup):
    pf, _ = pf_setup
    prior = np.array([0.0, 0.0, 1.0])  # all stress
    pf.initialise(prior)
    assert np.all(pf.particles == 2), "All particles should be in Stress"


def test_update_returns_valid_probs(pf_setup):
    pf, X = pf_setup
    pf.initialise()
    probs = pf.update(X[0])
    assert probs.shape == (3,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    assert (probs >= 0).all()


def test_update_batch_shape(pf_setup):
    pf, X = pf_setup
    pf.initialise()
    T = 50
    probs_seq = pf.update_batch(X[:T])
    assert probs_seq.shape == (T, 3)


def test_update_batch_probabilities_valid(pf_setup):
    pf, X = pf_setup
    pf.initialise()
    probs_seq = pf.update_batch(X[:100])
    row_sums = probs_seq.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    assert (probs_seq >= 0).all()


def test_ess_history_populated(pf_setup):
    pf, X = pf_setup
    pf.reset()
    pf.update_batch(X[:30])
    assert len(pf.ess_history) == 30


def test_map_state_valid(pf_setup):
    pf, X = pf_setup
    pf.initialise()
    pf.update(X[0])
    assert pf.get_map_state() in {0, 1, 2}


def test_reset_clears_history(pf_setup):
    pf, X = pf_setup
    pf.update_batch(X[:20])
    pf.reset()
    assert len(pf.ess_history) == 0
    assert len(pf.resample_flags) == 0


def test_convergence_stress_regime():
    """After many stress observations, P(Stress) should dominate."""
    from src.data.synthetic_generator import REGIME_PARAMS
    rng = np.random.default_rng(99)
    stress = REGIME_PARAMS[2]

    # Generate pure stress observations
    vols = np.array([stress.ig_vol, stress.hy_vol, stress.vix_vol])
    obs = rng.normal(
        loc=[stress.ig_mean / 100, stress.hy_mean / 100, stress.vix_mean / 10],
        scale=vols / 100,
        size=(200, 3),
    )

    A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.02, 0.08, 0.90]])
    means = np.array([
        [-1.0, -1.0, -1.0],
        [0.0,  0.0,  0.0],
        [2.0,  2.0,  2.0],  # stress has highest mean features
    ])
    covars = np.array([np.eye(3) * 0.1] * 3)

    pf = ParticleFilter(
        n_particles=1000, transition_matrix=A,
        means=means, covars=covars, seed=0
    )
    pf.initialise()
    for ob in obs:
        probs = pf.update(ob)

    assert probs[2] > 0.5, f"Stress prob {probs[2]:.3f} should exceed 0.5"


# ---------------------------------------------------------------------------
# Resampling tests
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_systematic_resample_count(rng):
    weights = np.ones(100) / 100
    idx = systematic_resample(weights, 100, rng)
    assert len(idx) == 100
    assert idx.min() >= 0
    assert idx.max() < 100


def test_stratified_resample_count(rng):
    weights = np.ones(100) / 100
    idx = stratified_resample(weights, 100, rng)
    assert len(idx) == 100


def test_resample_concentrates_on_high_weight(rng):
    weights = np.zeros(10)
    weights[7] = 1.0  # all weight on particle 7
    idx = systematic_resample(weights, 10, rng)
    assert np.all(idx == 7), "All resampled should be particle 7"


def test_systematic_vs_stratified_similar_output(rng):
    rng2 = np.random.default_rng(0)
    weights = np.array([0.4, 0.1, 0.3, 0.1, 0.1])
    idx_sys = systematic_resample(weights, 100, rng)
    idx_str = stratified_resample(weights, 100, rng2)
    # Both should heavily favour particle 0 and 2
    assert np.sum(idx_sys == 0) > 25
    assert np.sum(idx_str == 0) > 25
