# HMM Credit Regime Detection

**Hidden Markov Model & Bayesian Particle Filter for US Credit Market Regime Classification**

> Skills: Hidden Markov Models · Sequential Monte Carlo · Bayesian Filtering · US Corporate Bonds · Regime Switching · Streamlit

---

## Overview

Credit markets cycle through distinct regimes — tight-spread risk-on environments, normal conditions, and stress/widening periods. Identifying the current regime in real time is critical for position sizing, hedging, and execution timing.

This project builds a full regime detection system:

1. **HMM Training** — A 3-state Gaussian HMM (Baum-Welch / EM) trained on synthetic CDX IG/HY spread features, producing a transition matrix, per-state emission distributions, and initial state probabilities.
2. **Viterbi Decoding** — Offline most-likely state sequence for historical analysis and validation.
3. **Particle Filter** — A real-time Sequential Monte Carlo (SMC) filter with N=2,000 particles that updates the regime probability vector as each new observation arrives, without requiring Gaussian linearity assumptions.
4. **Signal Generation** — Continuous, probability-weighted trading signals: position scaling factor, execution urgency score, RFQ spread adjustment, and Kelly fraction — with no hard threshold switches.
5. **Streamlit Dashboard** — Four-tab interactive app including a live particle-filter simulation replay.

---

## Architecture

```
Raw Spread Data (CDX IG / HY / VIX)
        │
        ▼
Feature Engineering (9 features)
  ig_return, hy_return, roll_vol_21×2,
  hy_ig_spread, vix_level_norm,
  momentum_5×2, ig_autocorr_10
        │
        ├──► HMM (Baum-Welch)
        │        │
        │        ├──► Viterbi Path (offline)
        │        └──► Forward Probs (online)
        │
        └──► Particle Filter (N=2000, real-time)
                 │
                 ▼
         Regime Probability Vector
         [P(Tight), P(Normal), P(Stress)]
                 │
                 ▼
         Signal Generator
         ├── Position Scale  [0, 1]
         ├── Execution Urgency [0, 1]
         ├── Spread Adj (bps)
         └── Kelly Fraction
```

### Regime Definitions

| State | Label  | CDX IG (approx) | CDX HY (approx) | VIX    | Typical Duration |
|:-----:|--------|:-----------:|:-----------:|:------:|:----------------:|
| 0     | Tight  | ~60 bps     | ~270 bps    | ~13    | Persistent       |
| 1     | Normal | ~110 bps    | ~380 bps    | ~18    | Persistent       |
| 2     | Stress | ~210 bps    | ~650 bps    | ~32    | 20–60 days       |

---

## Project Structure

```
HMM Credit Regime Detection/
│
├── app.py                        # Streamlit dashboard (4 tabs)
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py   # Regime-switching CDX/VIX data generator
│   │   └── feature_builder.py       # 9-feature engineering pipeline
│   │
│   ├── models/
│   │   ├── hmm_model.py             # CreditHMM: Baum-Welch, BIC/AIC, state relabelling
│   │   └── viterbi.py               # NumPy Viterbi decoder + hmmlearn wrapper
│   │
│   ├── filters/
│   │   ├── particle_filter.py       # SMC particle filter (N=2000, ESS resampling)
│   │   └── resampling.py            # Systematic, stratified, multinomial resampling
│   │
│   ├── signals/
│   │   └── regime_signal.py         # SignalGenerator: position scale, urgency, Kelly
│   │
│   └── evaluation/
│       └── metrics.py               # Persistence, latency, Brier score, confusion matrix
│
├── tests/
│   ├── test_synthetic_generator.py  # 10 tests
│   ├── test_feature_builder.py      # 9 tests
│   ├── test_hmm_model.py            # 16 tests
│   ├── test_particle_filter.py      # 19 tests
│   └── test_signals.py              # 12 tests  (+ test_evaluation.py: 4 tests)
│                                    # Total: 70 tests, all passing
│
└── notebooks/
    ├── 01_model_selection.ipynb     # 2 vs 3 vs 4 states: BIC / AIC / log-likelihood
    ├── 02_regime_validation.ipynb   # Viterbi accuracy, confusion matrix, detection latency
    └── 03_signal_backtest.ipynb     # Regime-conditioned carry strategy backtest
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd "HMM Credit Regime Detection"
```

### 2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit Dashboard

```bash
.venv/bin/streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Workflow inside the dashboard:**
1. Adjust data / model / filter settings in the **sidebar**
2. Click **🚀 Train Model** — generates synthetic data, fits the HMM, runs Viterbi
3. Navigate to **🔴 Live Regime Monitor** → press **▶ Start** to replay the particle filter step-by-step
4. Explore **📈 Historical Analysis** for regime overlays on spreads
5. Inspect **🔬 Model Diagnostics** for the transition matrix heatmap and emission means
6. Review **📡 Signal Dashboard** for position scale, urgency, and Kelly fraction time series

### Run Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected output: **70 passed** in ~5 seconds.

### Run Notebooks

```bash
.venv/bin/jupyter notebook notebooks/
```

Open in order:
- `01_model_selection.ipynb` — justifies 3-state model choice
- `02_regime_validation.ipynb` — validates accuracy against ground truth
- `03_signal_backtest.ipynb` — out-of-sample Sharpe / drawdown analysis

---

## Key Design Decisions

### HMM State Relabelling
`hmmlearn` assigns states arbitrarily by internal initialisation. After fitting, states are remapped to a canonical order (Tight=0, Normal=1, Stress=2) by sorting emission means on the first feature (IG return). This ensures consistent interpretation across random restarts.

### Why Particle Filter on Top of HMM?
The HMM forward algorithm assumes Gaussian emissions and linear state transitions. The particle filter:
- Handles non-Gaussian or non-linear observation models via importance sampling
- Can integrate additional data sources (e.g., intra-day TRACE prints) without re-training
- Provides natural uncertainty quantification through the particle distribution
- Converges quickly (ESS-triggered systematic resampling avoids degeneracy)

### Probability-Weighted Signals
Rather than hard regime thresholds (e.g., "if P(Stress) > 0.5, halve position"), signals are continuous probability-weighted blends:

```
position_scale = P(Tight)×1.00 + P(Normal)×0.75 + P(Stress)×0.30
```

This eliminates discontinuous jumps as the regime probability crosses a threshold, which would cause excessive turnover in live trading.

---

## Validation Metrics

| Metric | Description |
|--------|-------------|
| Log-likelihood | Goodness of fit on training data |
| BIC / AIC | Model selection across 2, 3, 4 states |
| Regime persistence | Avg episode duration per state |
| Confusion matrix | Viterbi vs true synthetic labels |
| Brier score | Probability calibration quality |
| Detection latency | Avg days to detect stress onset |
| Out-of-sample Sharpe | Regime-scaled carry vs full-size baseline |

---

## Resume Bullet (Interview-Ready)

> Designed a Credit Market Regime Detection system combining a 3-state Gaussian Hidden Markov Model (Baum-Welch training on CDX IG/HY spreads) with a real-time Bayesian Particle Filter (N=2,000, systematic resampling) for continuous regime probability updating — translating probabilistic regime signals into dynamic position sizing and execution urgency scores for US corporate bond strategies.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `hmmlearn` | Gaussian HMM implementation (Baum-Welch EM) |
| `numpy` / `pandas` | Numerical computing, time series |
| `scipy` | Multivariate Gaussian pdfs, autocorrelation |
| `scikit-learn` | Classification metrics |
| `plotly` | Interactive charts (regime fan charts, heatmaps) |
| `streamlit` | Dashboard framework |
| `seaborn` / `matplotlib` | Notebook visualisations |
| `joblib` | Model serialisation |
| `pytest` | Test framework (70 tests) |
