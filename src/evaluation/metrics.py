"""
Evaluation metrics for regime detection quality.

Covers:
  - Regime persistence (average duration per state)
  - Transition frequency matrix
  - Classification accuracy vs manually labelled stress periods
  - Filter convergence speed (latency in detecting regime changes)
  - Log-likelihood, BIC, AIC for model selection
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# Regime persistence
# ---------------------------------------------------------------------------

def regime_persistence(states: np.ndarray, regime_names: list = None) -> pd.DataFrame:
    """
    Compute average duration (in periods) for each regime.

    Parameters
    ----------
    states : np.ndarray of int
    regime_names : list of str, optional

    Returns
    -------
    pd.DataFrame with columns [regime, n_episodes, avg_duration, min_dur, max_dur]
    """
    if regime_names is None:
        regime_names = ["Tight", "Normal", "Stress"]

    records = []
    for k, name in enumerate(regime_names):
        in_regime = (states == k).astype(int)
        # Find runs of 1s
        diff = np.diff(np.concatenate([[0], in_regime, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        durations = ends - starts

        records.append({
            "regime": name,
            "n_episodes": len(durations),
            "avg_duration": float(np.mean(durations)) if len(durations) else 0.0,
            "min_duration": int(np.min(durations)) if len(durations) else 0,
            "max_duration": int(np.max(durations)) if len(durations) else 0,
            "total_days": int(in_regime.sum()),
            "pct_time": float(in_regime.mean() * 100),
        })

    return pd.DataFrame(records).set_index("regime")


# ---------------------------------------------------------------------------
# Transition frequency
# ---------------------------------------------------------------------------

def transition_frequency(states: np.ndarray, n_states: int = 3) -> pd.DataFrame:
    """
    Empirical transition count matrix from a state sequence.
    Normalised by row sum to give empirical transition probabilities.
    """
    counts = np.zeros((n_states, n_states), dtype=int)
    for t in range(len(states) - 1):
        counts[states[t], states[t + 1]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, counts / row_sums, 0)

    names = ["Tight", "Normal", "Stress"][:n_states]
    return pd.DataFrame(probs, index=names, columns=names)


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------

def classification_accuracy(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    regime_names: list = None,
) -> dict:
    """
    Confusion matrix and per-class metrics between true and predicted regimes.

    Returns
    -------
    dict with keys: confusion_matrix (DataFrame), report (str), accuracy (float)
    """
    if regime_names is None:
        regime_names = ["Tight", "Normal", "Stress"]

    n = len(regime_names)
    cm = confusion_matrix(true_states, predicted_states, labels=list(range(n)))
    cm_df = pd.DataFrame(cm, index=regime_names, columns=regime_names)

    accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    report = classification_report(
        true_states, predicted_states,
        target_names=regime_names,
        zero_division=0,
    )

    return {
        "confusion_matrix": cm_df,
        "report": report,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Filter convergence latency
# ---------------------------------------------------------------------------

def detection_latency(
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    target_regime: int = 2,
    window: int = 5,
) -> float:
    """
    Average number of periods after a true regime change until the predicted
    state also switches to that regime.

    Parameters
    ----------
    target_regime : int   Regime to measure latency for (default 2 = Stress).
    window : int          Max look-ahead window in periods.

    Returns
    -------
    float : average detection latency in periods (NaN if no transitions found).
    """
    latencies = []
    T = len(true_states)

    for t in range(1, T - window):
        # Detect onset of target_regime in true states
        if true_states[t] == target_regime and true_states[t - 1] != target_regime:
            # Find first predicted match within window
            for lag in range(window + 1):
                if t + lag < T and predicted_states[t + lag] == target_regime:
                    latencies.append(lag)
                    break

    return float(np.mean(latencies)) if latencies else float("nan")


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

def brier_score(
    true_states: np.ndarray,
    probs: np.ndarray,
    n_classes: int = 3,
) -> float:
    """
    Multi-class Brier score: lower is better.

    Parameters
    ----------
    true_states : np.ndarray, shape (T,)
    probs : np.ndarray, shape (T, K)
    """
    T = len(true_states)
    one_hot = np.zeros((T, n_classes))
    for t, s in enumerate(true_states):
        one_hot[t, s] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
