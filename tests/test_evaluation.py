"""Tests for the evaluation metrics module."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    regime_persistence, transition_frequency, classification_accuracy,
    detection_latency, brier_score,
)


@pytest.fixture
def simple_states():
    # 10 Tight, 20 Normal, 15 Stress, 5 Normal
    return np.array(
        [0] * 10 + [1] * 20 + [2] * 15 + [1] * 5
    )


def test_persistence_shape(simple_states):
    df = regime_persistence(simple_states)
    assert set(df.index) == {"Tight", "Normal", "Stress"}
    assert "avg_duration" in df.columns


def test_persistence_tight_duration(simple_states):
    df = regime_persistence(simple_states)
    assert df.loc["Tight", "avg_duration"] == 10.0
    assert df.loc["Tight", "n_episodes"] == 1


def test_persistence_normal_multiple_episodes(simple_states):
    df = regime_persistence(simple_states)
    assert df.loc["Normal", "n_episodes"] == 2


def test_persistence_pct_sums_to_100(simple_states):
    df = regime_persistence(simple_states)
    total = df["pct_time"].sum()
    assert abs(total - 100.0) < 1e-6


def test_transition_frequency_shape(simple_states):
    freq = transition_frequency(simple_states, n_states=3)
    assert freq.shape == (3, 3)


def test_transition_frequency_row_stochastic(simple_states):
    freq = transition_frequency(simple_states, n_states=3)
    # Rows that have transitions should sum to 1
    for i in range(3):
        row_sum = freq.iloc[i].sum()
        assert abs(row_sum - 1.0) < 1e-10 or row_sum == 0.0


def test_classification_accuracy_perfect():
    states = np.array([0, 1, 2, 0, 1])
    result = classification_accuracy(states, states)
    assert result["accuracy"] == 1.0


def test_classification_accuracy_wrong():
    true = np.array([0, 0, 0, 1, 1])
    pred = np.array([1, 1, 1, 0, 0])
    result = classification_accuracy(true, pred)
    assert result["accuracy"] == 0.0


def test_classification_accuracy_confusion_matrix_shape():
    true = np.array([0, 1, 2, 0, 2])
    pred = np.array([0, 1, 1, 0, 2])
    result = classification_accuracy(true, pred)
    assert result["confusion_matrix"].shape == (3, 3)


def test_detection_latency_immediate():
    true =  np.array([1, 2, 2, 2, 2, 2])
    pred =  np.array([1, 2, 2, 2, 2, 2])
    lat = detection_latency(true, pred, target_regime=2)
    assert lat == 0.0


def test_detection_latency_delayed():
    true = np.array([1, 2, 2, 2, 2, 2, 2, 2])
    pred = np.array([1, 1, 1, 2, 2, 2, 2, 2])  # 2-period lag
    lat = detection_latency(true, pred, target_regime=2, window=5)
    assert lat == 2.0


def test_brier_score_perfect():
    true = np.array([0, 1, 2])
    probs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    bs = brier_score(true, probs)
    assert bs == 0.0


def test_brier_score_worst():
    true = np.array([0, 0, 0])
    probs = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
    bs = brier_score(true, probs)
    assert bs > 0


def test_brier_score_range():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 3, size=100)
    raw = rng.dirichlet([1, 1, 1], size=100)
    bs = brier_score(true, raw)
    assert 0.0 <= bs <= 2.0
