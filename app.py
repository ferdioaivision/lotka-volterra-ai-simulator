"""
Streamlit Application - Lotka-Volterra AI Simulator
-----------------------------------------------------
Interactive dashboard for real-time exploration of predator-prey dynamics.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.simulation import LotkaVolterraParams, simulate, generate_noisy_dataset

st.set_page_config(
    page_title="Lotka-Volterra AI Simulator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific look
st.markdown("""
<style>
    .main-title { font-size: 32px; font-weight: 600; color: #0E1117; }
    .section-header { font-size: 20px; font-weight: 500; margin-top: 20px; }
    .stSlider { padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Lotka-Volterra Population Dynamics Simulator</p>', unsafe_allow_html=True)
st.markdown("Mathematics & ML Project | Numerical Simulation with RK45")

# Sidebar - Parameters
st.sidebar.header("Model Parameters")
st.sidebar.markdown("Adjust the biological rates to observe dynamical changes.")

alpha = st.sidebar.slider("alpha - Prey growth rate", 0.1, 2.0, 1.1, 0.05, help="Intrinsic growth rate of prey in absence of predators")
beta = st.sidebar.slider("beta - Predation rate", 0.05, 1.0, 0.4, 0.05, help="Rate at which prey is consumed per predator-prey encounter")
delta = st.sidebar.slider("delta - Predator efficiency", 0.01, 0.5, 0.1, 0.01, help="Conversion efficiency of prey into predator reproduction")
gamma = st.sidebar.slider("gamma - Predator mortality", 0.1, 2.0, 0.4, 0.05, help="Mortality rate of predators in absence of prey")

st.sidebar.header("Initial Conditions")
x0 = st.sidebar.slider("Initial prey population", 5.0, 100.0, 40.0, 1.0)
y0 = st.sidebar.slider("Initial predator population", 2.0, 50.0, 9.0, 1.0)

st.sidebar.header("Simulation Settings")
t_end = st.sidebar.slider("Simulation duration", 10, 200, 100, 5)
noise = st.sidebar.slider("Observation noise level", 0.0, 0.20, 0.05, 0.01, help="Gaussian noise for AI dataset generation")

# Simulation execution
params = LotkaVolterraParams(alpha=alpha, beta=beta, delta=delta, gamma=gamma, x0=x0, y0=y0)

try:
    clean_df = simulate(params, t_span=(0, t_end), t_eval_points=1000)
    noisy_df = generate_noisy_dataset(clean_df, noise_level=noise)
except Exception as e:
    st.error(f"Simulation failed: {e}")
    st.stop()

# Layout - Two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">1. Temporal Evolution</p>', unsafe_allow_html=True)
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=clean_df["time"], y=clean_df["prey"], mode='lines', name='Prey (clean)', line=dict(color='#1f77b4', width=2)))
    fig_time.add_trace(go.Scatter(x=clean_df["time"], y=clean_df["predator"], mode='lines', name='Predator (clean)', line=dict(color='#d62728', width=2)))
    if noise > 0:
        fig_time.add_trace(go.Scatter(x=noisy_df["time"], y=noisy_df["prey_noisy"], mode='markers', name='Prey (noisy)', marker=dict(color='#1f77b4', size=3, opacity=0.4)))
        fig_time.add_trace(go.Scatter(x=noisy_df["time"], y=noisy_df["predator_noisy"], mode='markers', name='Predator (noisy)', marker=dict(color='#d62728', size=3, opacity=0.4)))
    
    fig_time.update_layout(
        xaxis_title="Time",
        yaxis_title="Population",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.markdown('<p class="section-header">2. Phase Portrait (x vs y)</p>', unsafe_allow_html=True)
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=clean_df["prey"], y=clean_df["predator"], mode='lines', name='Trajectory', line=dict(color='#2ca02c', width=2)))
    fig_phase.add_trace(go.Scatter(x=[clean_df["prey"].iloc[0]], y=[clean_df["predator"].iloc[0]], mode='markers', name='Initial state', marker=dict(color='black', size=10, symbol='circle-open', line=dict(width=2))))
    
    fig_phase.update_layout(
        xaxis_title="Prey population",
        yaxis_title="Predator population",
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# Data preview
st.markdown('<p class="section-header">3. Generated Dataset (for AI training)</p>', unsafe_allow_html=True)
st.dataframe(noisy_df.head(10), use_container_width=True)

# Download button
csv = noisy_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download dataset as CSV",
    data=csv,
    file_name="lotka_volterra_dataset.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("**Method:** `scipy.integrate.solve_ivp` with RK45 | **Tolerances:** rtol=1e-8, atol=1e-8 | **Model:** Classical Lotka-Volterra")
