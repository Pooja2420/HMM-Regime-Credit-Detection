"""
Viterbi decoder — finds the globally most likely state sequence.

Used offline for historical analysis and model validation.
The implementation is a transparent NumPy version that mirrors
the hmmlearn internals for educational clarity.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


def viterbi_decode(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Viterbi algorithm for a Gaussian HMM.

    Parameters
    ----------
    observations : np.ndarray, shape (T, d)
    pi : np.ndarray, shape (K,)         Initial state probabilities.
    A  : np.ndarray, shape (K, K)       Transition matrix (row-stochastic).
    means  : np.ndarray, shape (K, d)   Emission means per state.
    covars : np.ndarray, shape (K, d, d) Emission covariance matrices.

    Returns
    -------
    states : np.ndarray, shape (T,)     Most likely state sequence.
    log_prob : float                     Log-probability of that path.
    """
    T = len(observations)
    K = len(pi)

    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)

    # Precompute log emission probabilities
    log_emit = np.zeros((T, K))
    for k in range(K):
        rv = multivariate_normal(mean=means[k], cov=covars[k], allow_singular=True)
        log_emit[:, k] = rv.logpdf(observations)

    # Viterbi recursion
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    delta[0] = log_pi + log_emit[0]

    for t in range(1, T):
        for k in range(K):
            scores = delta[t - 1] + log_A[:, k]
            psi[t, k] = np.argmax(scores)
            delta[t, k] = scores[psi[t, k]] + log_emit[t, k]

    # Backtrack
    states = np.empty(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    log_prob = delta[-1, states[-1]]

    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states, log_prob


def viterbi_from_model(hmm_model, observations: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convenience wrapper using a fitted CreditHMM instance.
    Returns states in *canonical* label order.
    """
    params = hmm_model.get_emission_params()
    pi_raw = hmm_model.model.startprob_
    A_df = hmm_model.get_transition_matrix()

    # Convert canonical transition matrix back to numpy
    A = A_df.to_numpy()
    # Reorder pi to canonical
    pi = np.empty_like(pi_raw)
    for internal, canonical in enumerate(hmm_model._label_map):
        pi[canonical] = pi_raw[internal]

    return viterbi_decode(
        observations,
        pi,
        A,
        params["means"],
        params["covars"],
    )
