"""Tests for the signal generator."""
import numpy as np
import pandas as pd
import pytest

from src.signals.regime_signal import SignalGenerator, RegimeSignal, REGIME_NAMES


@pytest.fixture
def sig_gen():
    return SignalGenerator()


def test_generate_returns_regime_signal(sig_gen):
    probs = np.array([0.1, 0.7, 0.2])
    sig = sig_gen.generate(probs)
    assert isinstance(sig, RegimeSignal)


def test_generate_dominant_regime_normal(sig_gen):
    probs = np.array([0.1, 0.7, 0.2])
    sig = sig_gen.generate(probs)
    assert sig.dominant_regime == "Normal"


def test_generate_dominant_regime_stress(sig_gen):
    probs = np.array([0.05, 0.15, 0.80])
    sig = sig_gen.generate(probs)
    assert sig.dominant_regime == "Stress"


def test_position_scale_range(sig_gen):
    for probs in [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]:
        sig = sig_gen.generate(probs)
        assert 0.0 <= sig.position_scale <= 1.0


def test_stress_reduces_position(sig_gen):
    tight_sig = sig_gen.generate(np.array([1.0, 0.0, 0.0]))
    stress_sig = sig_gen.generate(np.array([0.0, 0.0, 1.0]))
    assert stress_sig.position_scale < tight_sig.position_scale


def test_stress_increases_urgency(sig_gen):
    tight_sig = sig_gen.generate(np.array([1.0, 0.0, 0.0]))
    stress_sig = sig_gen.generate(np.array([0.0, 0.0, 1.0]))
    assert stress_sig.execution_urgency > tight_sig.execution_urgency


def test_stress_increases_spread_adj(sig_gen):
    tight_sig = sig_gen.generate(np.array([1.0, 0.0, 0.0]))
    stress_sig = sig_gen.generate(np.array([0.0, 0.0, 1.0]))
    assert stress_sig.spread_adjustment_bps > tight_sig.spread_adjustment_bps


def test_kelly_fraction_range(sig_gen):
    probs = np.array([0.2, 0.5, 0.3])
    sig = sig_gen.generate(probs)
    assert 0.0 <= sig.kelly_fraction <= sig_gen.max_kelly


def test_generates_with_unnormalised_probs(sig_gen):
    probs = np.array([2.0, 3.0, 1.0])  # not normalised
    sig = sig_gen.generate(probs)
    assert abs(sig.tight_prob + sig.normal_prob + sig.stress_prob - 1.0) < 1e-6


def test_generate_series_shape(sig_gen):
    dates = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(0)
    raw = rng.dirichlet([1, 1, 1], size=100)
    df = pd.DataFrame(raw, index=dates, columns=["tight", "normal", "stress"])
    sig_df = sig_gen.generate_series(df)
    assert len(sig_df) == 100
    assert "position_scale" in sig_df.columns
    assert "execution_urgency" in sig_df.columns


def test_generate_series_position_scale_valid(sig_gen):
    dates = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(1)
    raw = rng.dirichlet([1, 1, 1], size=50)
    df = pd.DataFrame(raw, index=dates, columns=["tight", "normal", "stress"])
    sig_df = sig_gen.generate_series(df)
    assert (sig_df["position_scale"] >= 0).all()
    assert (sig_df["position_scale"] <= 1).all()
