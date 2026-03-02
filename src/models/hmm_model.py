"""
Gaussian HMM wrapper built on hmmlearn.

Provides training (Baum-Welch), state labelling, BIC/AIC model selection,
and persistence via joblib / MLflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical regime ordering: Tight=0, Normal=1, Stress=2
REGIME_ORDER = {"Tight": 0, "Normal": 1, "Stress": 2}
REGIME_NAMES = ["Tight", "Normal", "Stress"]


class CreditHMM:
    """
    3-state Gaussian HMM for credit market regime detection.

    Wraps hmmlearn.GaussianHMM and adds:
    - Automatic state relabelling (order by IG-return mean: low→Tight, high→Stress)
    - BIC / AIC for state-count selection
    - Forward algorithm for filtered state probabilities
    - Convenience fit / predict / score API
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
        n_init: int = 5,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.model: Optional[GaussianHMM] = None
        self._label_map: Optional[np.ndarray] = None   # internal_state -> canonical
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, lengths: Optional[list] = None) -> "CreditHMM":
        """
        Fit the HMM using Baum-Welch EM.  Runs n_init random restarts and
        keeps the model with highest log-likelihood.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
        lengths : list of int, optional
            Lengths for multiple sequences (defaults to [len(X)]).
        """
        if lengths is None:
            lengths = [len(X)]

        best_ll = -np.inf
        best_model = None

        for i in range(self.n_init):
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state + i,
                verbose=False,
            )
            try:
                model.fit(X, lengths)
                ll = model.score(X, lengths)
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception as e:
                logger.warning(f"HMM init {i} failed: {e}")

        if best_model is None:
            raise RuntimeError("All HMM initialisations failed.")

        self.model = best_model
        self._label_map = self._build_label_map(X)
        logger.info(f"HMM fitted. Log-likelihood={best_ll:.2f}")
        return self

    def _build_label_map(self, X: np.ndarray) -> np.ndarray:
        """
        Relabel internal states so that state 0=Tight, 1=Normal, 2=Stress.
        Ordering criterion: mean of feature[0] (ig_return proxy) ascending,
        OR if 3-state model, by mean of the first feature ascending.
        """
        if self.n_components != 3:
            return np.arange(self.n_components)

        means = self.model.means_[:, 0]  # first feature across states
        order = np.argsort(means)        # ascending: lowest mean = Tight
        label_map = np.empty(self.n_components, dtype=int)
        for canonical, internal in enumerate(order):
            label_map[internal] = canonical
        return label_map

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi-decoded state sequence, relabelled to canonical order."""
        raw = self.model.predict(X)
        return self._label_map[raw]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Forward-algorithm filtered state probabilities P(s_t | o_{1:t}).

        Returns
        -------
        np.ndarray, shape (T, n_components) — columns in canonical order.
        """
        _, posteriors = self.model.score_samples(X)
        # Reorder columns to canonical labelling
        reordered = np.empty_like(posteriors)
        for internal, canonical in enumerate(self._label_map):
            reordered[:, canonical] = posteriors[:, internal]
        return reordered

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of the observation sequence."""
        return self.model.score(X)

    # ------------------------------------------------------------------
    # Model selection metrics
    # ------------------------------------------------------------------

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion (lower is better)."""
        ll = self.score(X) * len(X)
        k = self._n_params()
        return -2 * ll + k * np.log(len(X))

    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion (lower is better)."""
        ll = self.score(X) * len(X)
        k = self._n_params()
        return -2 * ll + 2 * k

    def _n_params(self) -> int:
        n = self.n_components
        d = self.model.means_.shape[1]
        trans_params = n * (n - 1)
        mean_params = n * d
        if self.covariance_type == "full":
            cov_params = n * d * (d + 1) // 2
        elif self.covariance_type == "diag":
            cov_params = n * d
        else:
            cov_params = d * (d + 1) // 2
        return trans_params + mean_params + cov_params

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_transition_matrix(self) -> pd.DataFrame:
        """Transition matrix in canonical label order."""
        A_raw = self.model.transmat_
        n = self.n_components
        A_canon = np.empty((n, n))
        for i_raw in range(n):
            for j_raw in range(n):
                i_c = self._label_map[i_raw]
                j_c = self._label_map[j_raw]
                A_canon[i_c, j_c] = A_raw[i_raw, j_raw]
        names = REGIME_NAMES[:n] if n <= 3 else [str(i) for i in range(n)]
        return pd.DataFrame(A_canon, index=names, columns=names)

    def get_emission_params(self) -> dict:
        """Return means and covariances in canonical order."""
        n = self.n_components
        means = np.empty_like(self.model.means_)
        covs = np.empty_like(self.model.covars_)
        for internal, canonical in enumerate(self._label_map):
            means[canonical] = self.model.means_[internal]
            covs[canonical] = self.model.covars_[internal]
        return {"means": means, "covars": covs}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CreditHMM":
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
