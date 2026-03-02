"""
Regime-to-signal converter.

Translates a continuous regime probability vector into actionable outputs:
  - Position scaling factor (0–1 multiplier on base notional)
  - Execution urgency score (0–1; higher = more urgent to execute now)
  - Spread level adjustment for RFQ pricing (bps offset)
  - Risk regime label (most-likely state)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RegimeSignal:
    """Output of the signal generator for a single time step."""
    tight_prob: float
    normal_prob: float
    stress_prob: float
    dominant_regime: str
    position_scale: float      # [0, 1]: multiply base position size by this
    execution_urgency: float   # [0, 1]: 0=patient, 1=execute immediately
    spread_adjustment_bps: float  # RFQ bid/offer widening vs base
    kelly_fraction: float      # Suggested Kelly fraction given regime uncertainty

    @property
    def probs(self) -> np.ndarray:
        return np.array([self.tight_prob, self.normal_prob, self.stress_prob])


# ---------------------------------------------------------------------------
# Per-regime calibration constants
# ---------------------------------------------------------------------------

# Position scale: full size in tight, reduced in normal, minimal in stress
POSITION_SCALE = {
    "Tight": 1.00,
    "Normal": 0.75,
    "Stress": 0.30,
}

# Execution urgency: in stress you want to execute quickly (market dries up)
EXECUTION_URGENCY = {
    "Tight": 0.20,
    "Normal": 0.50,
    "Stress": 0.90,
}

# Spread adjustment (bps to add to mid for RFQ offer price)
SPREAD_ADJUSTMENT_BPS = {
    "Tight": 2.0,
    "Normal": 5.0,
    "Stress": 15.0,
}

REGIME_NAMES = ["Tight", "Normal", "Stress"]


class SignalGenerator:
    """
    Converts regime probability vector to trading signals.

    Signals are probability-weighted blends of per-regime calibrations,
    so they vary continuously as regime probabilities shift — no hard
    threshold switches.
    """

    def __init__(
        self,
        position_scale: dict = None,
        execution_urgency: dict = None,
        spread_adj_bps: dict = None,
        max_kelly: float = 0.25,
    ):
        self.pos_scale = position_scale or POSITION_SCALE
        self.exec_urgency = execution_urgency or EXECUTION_URGENCY
        self.spread_adj = spread_adj_bps or SPREAD_ADJUSTMENT_BPS
        self.max_kelly = max_kelly

    def generate(self, regime_probs: np.ndarray) -> RegimeSignal:
        """
        Generate signal from a 3-element probability vector.

        Parameters
        ----------
        regime_probs : np.ndarray, shape (3,)
            [P(Tight), P(Normal), P(Stress)]

        Returns
        -------
        RegimeSignal
        """
        p = np.clip(regime_probs, 0, 1)
        p /= p.sum()  # renormalise

        tight, normal, stress = p

        # Dominant regime
        dominant_idx = int(np.argmax(p))
        dominant = REGIME_NAMES[dominant_idx]

        # Probability-weighted signals
        pos_vals = np.array([self.pos_scale[r] for r in REGIME_NAMES])
        urg_vals = np.array([self.exec_urgency[r] for r in REGIME_NAMES])
        sprd_vals = np.array([self.spread_adj[r] for r in REGIME_NAMES])

        position_scale = float(np.dot(p, pos_vals))
        execution_urgency = float(np.dot(p, urg_vals))
        spread_adj_bps = float(np.dot(p, sprd_vals))

        # Kelly fraction: scale by position_scale, penalise uncertainty
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        max_entropy = np.log(len(p))
        uncertainty_penalty = 1.0 - entropy / max_entropy
        kelly_fraction = self.max_kelly * position_scale * uncertainty_penalty

        return RegimeSignal(
            tight_prob=float(tight),
            normal_prob=float(normal),
            stress_prob=float(stress),
            dominant_regime=dominant,
            position_scale=position_scale,
            execution_urgency=execution_urgency,
            spread_adjustment_bps=spread_adj_bps,
            kelly_fraction=kelly_fraction,
        )

    def generate_series(self, probs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for a time series of regime probabilities.

        Parameters
        ----------
        probs_df : pd.DataFrame
            Columns: ['tight', 'normal', 'stress'] or matching REGIME_NAMES.
            Index: DatetimeIndex.

        Returns
        -------
        pd.DataFrame with signal columns indexed like probs_df.
        """
        records = []
        for _, row in probs_df.iterrows():
            sig = self.generate(row.to_numpy())
            records.append({
                "dominant_regime": sig.dominant_regime,
                "position_scale": sig.position_scale,
                "execution_urgency": sig.execution_urgency,
                "spread_adj_bps": sig.spread_adjustment_bps,
                "kelly_fraction": sig.kelly_fraction,
            })
        return pd.DataFrame(records, index=probs_df.index)
