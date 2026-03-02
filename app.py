"""
HMM Credit Regime Detection — Streamlit Dashboard
==================================================
Pages:
  1. Live Regime Monitor   — real-time particle filter simulation
  2. Historical Analysis   — regime overlay on full spread history
  3. Model Diagnostics     — transition matrix, emission distributions, BIC/AIC
  4. Signal Dashboard      — position scaling, execution urgency, Kelly sizing
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Project imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.synthetic_generator import (
    generate_credit_data, get_regime_colors, get_regime_labels,
    DEFAULT_TRANSITION, REGIME_NAMES,
)
from src.data.feature_builder import build_features, standardise_features, get_observation_matrix
from src.models.hmm_model import CreditHMM
from src.models.viterbi import viterbi_from_model
from src.filters.particle_filter import ParticleFilter
from src.signals.regime_signal import SignalGenerator
from src.evaluation.metrics import (
    regime_persistence, transition_frequency, classification_accuracy,
    detection_latency, brier_score,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HMM Credit Regime Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGIME_COLORS = get_regime_colors()
REGIME_LABELS = get_regime_labels()
REGIME_PALETTE = ["#2ecc71", "#3498db", "#e74c3c"]

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session_state():
    defaults = {
        "model_trained": False,
        "hmm": None,
        "raw_df": None,
        "feat_df": None,
        "X_train": None,
        "X_all": None,
        "scaler": None,
        "viterbi_states": None,
        "hmm_probs": None,
        "pf_probs": None,
        "sim_running": False,
        "sim_step": 0,
        "sim_probs": [],
        "sim_ess": [],
        "signal_gen": SignalGenerator(),
        "n_particles": 2000,
        "seed": 42,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session_state()

# ---------------------------------------------------------------------------
# Sidebar — data & model controls
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuration")

with st.sidebar.expander("Data Settings", expanded=True):
    n_days = st.slider("Simulation length (days)", 500, 5000, 3500, 250)
    seed = st.number_input("Random seed", 0, 9999, 42, 1)
    train_frac = st.slider("Train / test split", 0.5, 0.9, 0.75, 0.05)

with st.sidebar.expander("HMM Settings"):
    n_states = st.selectbox("Number of states", [2, 3, 4], index=1)
    cov_type = st.selectbox("Covariance type", ["full", "diag", "spherical"], index=0)
    n_init = st.slider("Random restarts (n_init)", 1, 10, 5)

with st.sidebar.expander("Particle Filter Settings"):
    n_particles = st.select_slider(
        "N particles", [250, 500, 1000, 2000, 5000], value=2000
    )
    resample_method = st.selectbox("Resampling", ["systematic", "stratified"])
    ess_threshold = st.slider("ESS threshold (fraction of N)", 0.1, 0.9, 0.5, 0.05)

if st.sidebar.button("🚀 Train Model", type="primary"):
    with st.spinner("Generating data and training HMM..."):
        # Generate data
        df, true_states = generate_credit_data(n_days=n_days, seed=int(seed))
        feat = build_features(df)

        # Align true regimes to feature index
        true_aligned = true_states[df.index.isin(feat.index)]

        # Train/test split
        n_train = int(len(feat) * train_frac)
        feat_train = feat.iloc[:n_train]
        feat_test = feat.iloc[n_train:]

        feat_train_s, feat_test_s, scaler = standardise_features(feat_train, feat_test)
        X_train = get_observation_matrix(feat_train_s)
        X_all = get_observation_matrix(
            pd.concat([feat_train_s, feat_test_s])
        )

        # Train HMM
        hmm = CreditHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_init=n_init,
            random_state=int(seed),
        )
        hmm.feature_names = feat.columns.tolist()
        hmm.fit(X_train)

        # Viterbi decode full series
        viterbi_states, _ = viterbi_from_model(hmm, X_all)

        # HMM filtered probs (forward algorithm)
        hmm_probs = hmm.predict_proba(X_all)

        # Store in session state
        feat_full = pd.concat([feat_train_s, feat_test_s])
        st.session_state.update({
            "model_trained": True,
            "hmm": hmm,
            "raw_df": df,
            "feat_df": feat_full,
            "X_train": X_train,
            "X_all": X_all,
            "scaler": scaler,
            "true_states": true_aligned,
            "viterbi_states": viterbi_states,
            "hmm_probs": hmm_probs,
            "pf_probs": None,  # computed on demand
            "n_particles": n_particles,
            "seed": int(seed),
            "feat_index": feat_full.index,
            "raw_df_aligned": df.loc[feat_full.index],
        })

    st.sidebar.success("Model trained!")

# ---------------------------------------------------------------------------
# Page tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Regime Monitor",
    "📈 Historical Analysis",
    "🔬 Model Diagnostics",
    "📡 Signal Dashboard",
])


# ===========================================================================
# TAB 1 — Live Regime Monitor
# ===========================================================================

with tab1:
    st.header("Live Regime Monitor")
    st.markdown(
        "Replay the particle filter step-by-step to simulate real-time regime detection. "
        "Each frame updates the regime probability fan chart and ESS trace."
    )

    if not st.session_state.model_trained:
        st.info("Train the model using the sidebar first.")
    else:
        hmm = st.session_state.hmm
        X_all = st.session_state.X_all
        feat_idx = st.session_state.feat_index

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            sim_speed = st.select_slider(
                "Simulation speed", ["Slow", "Normal", "Fast", "Max"],
                value="Normal"
            )
        speed_map = {"Slow": 0.15, "Normal": 0.05, "Fast": 0.01, "Max": 0.0}
        delay = speed_map[sim_speed]

        with col_ctrl2:
            start_btn = st.button("▶ Start / Resume", type="primary")
        with col_ctrl3:
            reset_btn = st.button("⏮ Reset")

        if reset_btn:
            st.session_state.sim_step = 0
            st.session_state.sim_probs = []
            st.session_state.sim_ess = []
            st.session_state.sim_running = False

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Placeholder charts
        prob_chart_ph = st.empty()
        ess_chart_ph = st.empty()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        tight_metric = col_m1.empty()
        normal_metric = col_m2.empty()
        stress_metric = col_m3.empty()
        map_metric = col_m4.empty()

        if start_btn and st.session_state.sim_step < len(X_all):
            # Initialise particle filter
            params = hmm.get_emission_params()
            A_df = hmm.get_transition_matrix()
            n_comp = hmm.n_components

            pf = ParticleFilter(
                n_particles=st.session_state.n_particles,
                transition_matrix=A_df.to_numpy(),
                means=params["means"][:n_comp],
                covars=params["covars"][:n_comp],
                resample_method=resample_method,
                ess_threshold=ess_threshold,
                seed=st.session_state.seed,
            )
            # Restore prior steps if resuming
            if st.session_state.sim_step > 0:
                prior_probs = np.array(st.session_state.sim_probs[-1])
                pf.initialise(prior_probs)
                # fast-forward (no rendering)
                for t in range(st.session_state.sim_step):
                    pf.update(X_all[t])
            else:
                pf.initialise()

            T = len(X_all)
            st.session_state.sim_running = True

            for t in range(st.session_state.sim_step, T):
                probs = pf.update(X_all[t])
                ess = pf.ess_history[-1]

                st.session_state.sim_probs.append(probs.tolist())
                st.session_state.sim_ess.append(ess)
                st.session_state.sim_step = t + 1

                # Update UI every 10 steps
                if t % 10 == 0 or t == T - 1:
                    pct = (t + 1) / T
                    progress_bar.progress(pct)
                    status_text.text(
                        f"Step {t+1}/{T} — {feat_idx[t].strftime('%Y-%m-%d')}"
                    )

                    probs_arr = np.array(st.session_state.sim_probs)
                    ess_arr = np.array(st.session_state.sim_ess)
                    idx_so_far = feat_idx[:t+1]

                    # --- Regime probability fan chart ---
                    fig_prob = go.Figure()
                    n_comp_local = probs_arr.shape[1]
                    labels = REGIME_NAMES[:n_comp_local]
                    colors = REGIME_PALETTE[:n_comp_local]
                    for k in range(n_comp_local):
                        fig_prob.add_trace(go.Scatter(
                            x=idx_so_far,
                            y=probs_arr[:, k],
                            name=labels[k],
                            mode="lines",
                            line=dict(color=colors[k], width=2),
                            stackgroup="one",
                            fillcolor=colors[k],
                        ))
                    fig_prob.update_layout(
                        title="Regime Probabilities (Particle Filter)",
                        yaxis_title="P(regime)",
                        yaxis=dict(range=[0, 1]),
                        height=320,
                        margin=dict(t=40, b=20),
                        legend=dict(orientation="h", y=-0.15),
                    )
                    prob_chart_ph.plotly_chart(fig_prob, use_container_width=True)

                    # --- ESS trace ---
                    fig_ess = go.Figure(go.Scatter(
                        x=idx_so_far,
                        y=ess_arr / st.session_state.n_particles,
                        mode="lines",
                        line=dict(color="#9b59b6", width=1.5),
                        name="N_eff / N",
                    ))
                    fig_ess.add_hline(
                        y=ess_threshold,
                        line_dash="dot", line_color="red",
                        annotation_text=f"Resample threshold ({ess_threshold:.0%})"
                    )
                    fig_ess.update_layout(
                        title="Effective Sample Size",
                        yaxis_title="N_eff / N",
                        yaxis=dict(range=[0, 1]),
                        height=200,
                        margin=dict(t=40, b=20),
                    )
                    ess_chart_ph.plotly_chart(fig_ess, use_container_width=True)

                    # --- Metrics ---
                    latest = probs
                    map_state = int(np.argmax(latest))
                    tight_metric.metric("P(Tight)", f"{latest[0]:.1%}")
                    if n_comp_local >= 2:
                        normal_metric.metric("P(Normal)", f"{latest[1]:.1%}")
                    if n_comp_local >= 3:
                        stress_metric.metric("P(Stress)", f"{latest[2]:.1%}")
                    map_metric.metric("MAP State", REGIME_NAMES[map_state])

                    if delay > 0:
                        time.sleep(delay)

            st.session_state.sim_running = False
            st.success("Simulation complete!")

        elif st.session_state.sim_step > 0:
            # Show last computed state
            probs_arr = np.array(st.session_state.sim_probs)
            ess_arr = np.array(st.session_state.sim_ess)
            idx_so_far = feat_idx[:st.session_state.sim_step]

            fig_prob = go.Figure()
            for k in range(probs_arr.shape[1]):
                fig_prob.add_trace(go.Scatter(
                    x=idx_so_far, y=probs_arr[:, k],
                    name=REGIME_NAMES[k], mode="lines",
                    line=dict(color=REGIME_PALETTE[k], width=2),
                    stackgroup="one", fillcolor=REGIME_PALETTE[k],
                ))
            fig_prob.update_layout(
                title="Regime Probabilities (Particle Filter)",
                yaxis=dict(range=[0, 1]), height=320,
                margin=dict(t=40, b=20),
                legend=dict(orientation="h", y=-0.15),
            )
            prob_chart_ph.plotly_chart(fig_prob, use_container_width=True)

            latest = probs_arr[-1]
            tight_metric.metric("P(Tight)", f"{latest[0]:.1%}")
            if probs_arr.shape[1] >= 2:
                normal_metric.metric("P(Normal)", f"{latest[1]:.1%}")
            if probs_arr.shape[1] >= 3:
                stress_metric.metric("P(Stress)", f"{latest[2]:.1%}")
            map_metric.metric("MAP State", REGIME_NAMES[int(np.argmax(latest))])


# ===========================================================================
# TAB 2 — Historical Analysis
# ===========================================================================

with tab2:
    st.header("Historical Regime Analysis")

    if not st.session_state.model_trained:
        st.info("Train the model using the sidebar first.")
    else:
        raw = st.session_state.raw_df_aligned
        hmm_probs = st.session_state.hmm_probs
        viterbi_states = st.session_state.viterbi_states
        true_states = st.session_state.true_states
        feat_idx = st.session_state.feat_index
        n_comp = st.session_state.hmm.n_components

        view_mode = st.radio(
            "State sequence source",
            ["Viterbi (offline)", "HMM Forward (filtered)", "True Labels"],
            horizontal=True,
        )

        if view_mode == "Viterbi (offline)":
            display_states = viterbi_states
        elif view_mode == "HMM Forward (filtered)":
            display_states = np.argmax(hmm_probs, axis=1)
        else:
            display_states = true_states

        # --- Spread series with regime shading ---
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.4, 0.2],
            subplot_titles=["CDX IG Spread (bps)", "CDX HY Spread (bps)", "VIX"],
        )

        for series, row, color in [
            (raw["cdx_ig"], 1, "#2980b9"),
            (raw["cdx_hy"], 2, "#8e44ad"),
            (raw["vix"], 3, "#e67e22"),
        ]:
            fig.add_trace(
                go.Scatter(x=feat_idx, y=series.loc[feat_idx],
                           mode="lines", line=dict(color=color, width=1.5),
                           showlegend=False),
                row=row, col=1
            )

        # Regime background shading
        legend_added = set()
        for t in range(len(feat_idx) - 1):
            regime = int(display_states[t])
            rname = REGIME_NAMES[regime] if regime < len(REGIME_NAMES) else str(regime)
            rcolor = REGIME_PALETTE[regime] if regime < 3 else "#cccccc"
            show = rname not in legend_added
            for row in range(1, 4):
                fig.add_vrect(
                    x0=feat_idx[t], x1=feat_idx[t + 1],
                    fillcolor=rcolor, opacity=0.15,
                    layer="below", line_width=0,
                    row=row, col=1,
                    annotation_text=rname if (show and row == 1) else "",
                    annotation_position="top left",
                )
            if show:
                legend_added.add(rname)

        fig.update_layout(
            height=600,
            title_text=f"Regime Overlay — {view_mode}",
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Regime statistics ---
        st.subheader("Regime Persistence Statistics")
        n_names = REGIME_NAMES[:n_comp]
        persist_df = regime_persistence(display_states, n_names)
        st.dataframe(persist_df.style.format({
            "avg_duration": "{:.1f}",
            "pct_time": "{:.1f}%",
        }))

        # --- Classification vs true labels ---
        st.subheader("Classification Accuracy vs True Regimes")
        acc_result = classification_accuracy(true_states, display_states, n_names)
        col_a, col_b = st.columns([1, 2])
        col_a.metric("Accuracy", f"{acc_result['accuracy']:.1%}")
        col_b.text(acc_result["report"])

        cm_df = acc_result["confusion_matrix"]
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        fig_cm.update_layout(height=350, margin=dict(t=50))
        st.plotly_chart(fig_cm, use_container_width=True)


# ===========================================================================
# TAB 3 — Model Diagnostics
# ===========================================================================

with tab3:
    st.header("Model Diagnostics")

    if not st.session_state.model_trained:
        st.info("Train the model using the sidebar first.")
    else:
        hmm = st.session_state.hmm
        X_all = st.session_state.X_all
        X_train = st.session_state.X_train
        n_comp = hmm.n_components

        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Log-Likelihood (train)", f"{hmm.score(X_train):.4f}")
        col_d2.metric("BIC", f"{hmm.bic(X_train):.1f}")
        col_d3.metric("AIC", f"{hmm.aic(X_train):.1f}")

        # --- Transition matrix heatmap ---
        st.subheader("Transition Matrix")
        trans_df = hmm.get_transition_matrix()
        fig_trans = px.imshow(
            trans_df,
            text_auto=".3f",
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            title="Estimated Transition Probabilities P(s_t | s_{t-1})",
        )
        fig_trans.update_layout(height=350, margin=dict(t=50))
        st.plotly_chart(fig_trans, use_container_width=True)

        # --- Emission distributions ---
        st.subheader("Emission Means (per state, per feature)")
        params = hmm.get_emission_params()
        names = REGIME_NAMES[:n_comp]
        feat_names = hmm.feature_names if hmm.feature_names else [
            f"f{i}" for i in range(params["means"].shape[1])
        ]
        means_df = pd.DataFrame(
            params["means"], index=names, columns=feat_names
        )
        fig_means = px.imshow(
            means_df,
            text_auto=".3f",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Emission Mean Vectors (standardised feature space)",
        )
        fig_means.update_layout(height=350, margin=dict(t=50))
        st.plotly_chart(fig_means, use_container_width=True)

        # --- Empirical vs estimated transition ---
        st.subheader("Empirical Transition (Viterbi decoded)")
        viterbi_states = st.session_state.viterbi_states
        emp_trans = transition_frequency(viterbi_states, n_states=n_comp)
        fig_emp = px.imshow(
            emp_trans,
            text_auto=".3f",
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            title="Empirical Transition Frequencies from Viterbi Path",
        )
        fig_emp.update_layout(height=350, margin=dict(t=50))
        st.plotly_chart(fig_emp, use_container_width=True)

        # --- Brier score ---
        true_states = st.session_state.true_states
        hmm_probs = st.session_state.hmm_probs
        if len(hmm_probs) == len(true_states):
            bs = brier_score(true_states, hmm_probs, n_classes=n_comp)
            st.metric("Brier Score (HMM forward)", f"{bs:.4f}")

        # --- Detection latency ---
        viterbi_states = st.session_state.viterbi_states
        lat = detection_latency(true_states, viterbi_states, target_regime=2)
        st.metric(
            "Avg Stress Detection Latency (Viterbi)",
            f"{lat:.1f} days" if not np.isnan(lat) else "N/A"
        )


# ===========================================================================
# TAB 4 — Signal Dashboard
# ===========================================================================

with tab4:
    st.header("Signal Dashboard")

    if not st.session_state.model_trained:
        st.info("Train the model using the sidebar first.")
    else:
        hmm_probs = st.session_state.hmm_probs
        feat_idx = st.session_state.feat_index
        sig_gen = st.session_state.signal_gen
        n_comp = st.session_state.hmm.n_components

        # Pad columns to 3 if n_comp < 3
        probs_padded = np.zeros((len(hmm_probs), 3))
        probs_padded[:, :n_comp] = hmm_probs

        prob_df = pd.DataFrame(
            probs_padded,
            index=feat_idx,
            columns=["tight", "normal", "stress"],
        )
        signals_df = sig_gen.generate_series(prob_df)

        # --- Summary metrics at last timestep ---
        last_sig = sig_gen.generate(probs_padded[-1])
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Current Regime", last_sig.dominant_regime)
        col_s2.metric("Position Scale", f"{last_sig.position_scale:.2f}×")
        col_s3.metric("Execution Urgency", f"{last_sig.execution_urgency:.2f}")
        col_s4.metric("Spread Adj (bps)", f"{last_sig.spread_adjustment_bps:.1f}")

        # --- Signal time series ---
        fig_sig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=[
                "Position Scaling Factor",
                "Execution Urgency Score",
                "RFQ Spread Adjustment (bps)",
            ],
        )
        for (col, row, color) in [
            ("position_scale", 1, "#27ae60"),
            ("execution_urgency", 2, "#e74c3c"),
            ("spread_adj_bps", 3, "#9b59b6"),
        ]:
            fig_sig.add_trace(
                go.Scatter(
                    x=feat_idx,
                    y=signals_df[col],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=col.replace("_", " ").title(),
                    showlegend=True,
                ),
                row=row, col=1,
            )
        fig_sig.update_layout(
            height=500, title="Regime-Derived Trading Signals",
            margin=dict(t=60, b=20),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # --- Kelly fraction ---
        st.subheader("Kelly Fraction (Regime-Adjusted)")
        fig_kelly = go.Figure(go.Scatter(
            x=feat_idx,
            y=signals_df["kelly_fraction"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#f39c12", width=1.5),
            name="Kelly Fraction",
        ))
        fig_kelly.update_layout(
            yaxis_title="Suggested Kelly Fraction",
            height=250,
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig_kelly, use_container_width=True)

        # --- Signal table (last 20 rows) ---
        st.subheader("Signal Log (latest 20 observations)")
        display_sig = signals_df.tail(20).copy()
        display_sig.index = display_sig.index.strftime("%Y-%m-%d")
        st.dataframe(
            display_sig.style.format({
                "position_scale": "{:.3f}",
                "execution_urgency": "{:.3f}",
                "spread_adj_bps": "{:.2f}",
                "kelly_fraction": "{:.4f}",
            }).background_gradient(
                subset=["execution_urgency"], cmap="RdYlGn_r"
            )
        )
