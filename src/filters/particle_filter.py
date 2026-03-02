"""
Sequential Monte Carlo (SMC) Particle Filter for real-time regime tracking.

At each time step:
  1. Propagate particles through the HMM transition matrix.
  2. Reweight by Gaussian likelihood of the new observation.
  3. Normalise weights.
  4. Resample when effective sample size falls below threshold.
  5. Output empirical regime probability vector.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from .resampling import systematic_resample, stratified_resample


class ParticleFilter:
    """
    Particle filter for a discrete-state HMM with Gaussian emissions.

    Parameters
    ----------
    n_particles : int
        Number of particles (default 2000).
    transition_matrix : np.ndarray, shape (K, K)
        Row-stochastic HMM transition matrix.
    means : np.ndarray, shape (K, d)
        Emission mean vectors per state.
    covars : np.ndarray, shape (K, d, d)
        Emission covariance matrices per state.
    resample_method : str
        'systematic' (default) or 'stratified'.
    ess_threshold : float
        Fraction of N_particles below which resampling triggers (default 0.5).
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        n_particles: int = 2000,
        transition_matrix: np.ndarray = None,
        means: np.ndarray = None,
        covars: np.ndarray = None,
        resample_method: str = "systematic",
        ess_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.N = n_particles
        self.A = transition_matrix
        self.means = means
        self.covars = covars
        self.resample_method = resample_method
        self.ess_threshold = ess_threshold
        self.rng = np.random.default_rng(seed)

        self.n_states = len(transition_matrix)
        self._resample_fn = (
            systematic_resample
            if resample_method == "systematic"
            else stratified_resample
        )

        # Particle state (integer regime indices) and weights
        self.particles: np.ndarray = None
        self.weights: np.ndarray = None
        self._initialised = False

        # Diagnostics
        self.ess_history: list[float] = []
        self.resample_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, prior: np.ndarray = None) -> None:
        """
        Draw initial particles from prior distribution over states.

        Parameters
        ----------
        prior : np.ndarray, shape (K,), optional
            Initial state probabilities (uniform if None).
        """
        if prior is None:
            prior = np.ones(self.n_states) / self.n_states

        self.particles = self.rng.choice(
            self.n_states, size=self.N, p=prior
        )
        self.weights = np.ones(self.N) / self.N
        self._initialised = True

    # ------------------------------------------------------------------
    # Core update step
    # ------------------------------------------------------------------

    def update(self, observation: np.ndarray) -> np.ndarray:
        """
        Process one new observation and return updated regime probabilities.

        Parameters
        ----------
        observation : np.ndarray, shape (d,)
            Feature vector for the current time step.

        Returns
        -------
        regime_probs : np.ndarray, shape (K,)
            Weighted empirical distribution P(regime) over states.
        """
        if not self._initialised:
            self.initialise()

        # 1. Propagate: sample next state from transition distribution
        self.particles = self._propagate(self.particles)

        # 2. Reweight by emission likelihood
        likelihoods = self._emission_likelihood(observation, self.particles)
        self.weights = self.weights * likelihoods
        self.weights += 1e-300  # avoid zero weights

        # 3. Normalise
        self.weights /= self.weights.sum()

        # 4. Compute ESS and conditionally resample
        ess = self._effective_sample_size()
        self.ess_history.append(ess)
        resampled = False
        if ess < self.ess_threshold * self.N:
            indices = self._resample_fn(self.weights, self.N, self.rng)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N) / self.N
            resampled = True
        self.resample_flags.append(resampled)

        # 5. Compute empirical regime probability
        return self._regime_probabilities()

    def update_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Process a sequence of observations.

        Parameters
        ----------
        observations : np.ndarray, shape (T, d)

        Returns
        -------
        probs_sequence : np.ndarray, shape (T, K)
        """
        if not self._initialised:
            self.initialise()

        T = len(observations)
        probs = np.zeros((T, self.n_states))
        for t in range(T):
            probs[t] = self.update(observations[t])
        return probs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, particles: np.ndarray) -> np.ndarray:
        """Vectorised propagation: draw next state from A[current_state]."""
        new_particles = np.empty_like(particles)
        for k in range(self.n_states):
            mask = particles == k
            n_k = mask.sum()
            if n_k > 0:
                new_particles[mask] = self.rng.choice(
                    self.n_states, size=n_k, p=self.A[k]
                )
        return new_particles

    def _emission_likelihood(
        self, observation: np.ndarray, particles: np.ndarray
    ) -> np.ndarray:
        """
        Compute p(o_t | s_t^(i)) for each particle.
        Uses pre-computed per-state multivariate Gaussian pdfs.
        """
        likelihoods = np.zeros(self.N)
        for k in range(self.n_states):
            mask = particles == k
            if mask.sum() > 0:
                rv = multivariate_normal(
                    mean=self.means[k], cov=self.covars[k], allow_singular=True
                )
                likelihoods[mask] = rv.pdf(observation)
        return likelihoods

    def _effective_sample_size(self) -> float:
        """N_eff = 1 / sum(w_i^2)."""
        return 1.0 / np.sum(self.weights ** 2)

    def _regime_probabilities(self) -> np.ndarray:
        """Weighted fraction of particles in each state."""
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            probs[k] = self.weights[self.particles == k].sum()
        return probs

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_map_state(self) -> int:
        """Return the maximum a-posteriori regime."""
        return int(np.argmax(self._regime_probabilities()))

    def reset(self, prior: np.ndarray = None) -> None:
        """Reset particles and history (useful for fresh simulation runs)."""
        self.ess_history.clear()
        self.resample_flags.clear()
        self.initialise(prior)
