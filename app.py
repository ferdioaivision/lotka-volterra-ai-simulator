"""
Streamlit Application - Lotka-Volterra AI Simulator
---------------------------------------------------
Three tabs: interactive simulation, one-step-ahead machine learning, and the
fit of the model to the Hudson Bay lynx-hare data.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ai_model import results_to_frame, train_and_evaluate, unseen_orbit_evaluation
from src.real_data_validation import (fit_lotka_volterra, holdout_check,
                                      load_real_data, simulate_fit)
from src.simulation import LotkaVolterraParams, generate_noisy_dataset, invariant_drift, simulate

st.set_page_config(page_title="Lotka-Volterra AI Simulator", layout="wide", initial_sidebar_state="expanded")

st.title("Lotka-Volterra Population Dynamics Simulator")
st.caption("Mathematics & ML project | Numerical simulation with RK45, one-step prediction, real-data fit")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Model Parameters")
st.sidebar.markdown("Adjust the biological rates to observe dynamical changes.")

alpha = st.sidebar.slider("alpha - Prey growth rate", 0.1, 2.0, 1.1, 0.05, help="Intrinsic growth rate of prey in absence of predators")
beta = st.sidebar.slider("beta - Predation rate", 0.05, 1.0, 0.4, 0.05, help="Rate at which prey is consumed per predator-prey encounter")
delta = st.sidebar.slider("delta - Predator efficiency", 0.01, 0.5, 0.1, 0.01, help="Conversion efficiency of prey into predator reproduction")
gamma = st.sidebar.slider("gamma - Predator mortality", 0.1, 2.0, 0.4, 0.05, help="Mortality rate of predators in absence of prey")

st.sidebar.header("Initial Conditions")
x0 = st.sidebar.slider("Initial prey population", 5.0, 100.0, 10.0, 1.0)
y0 = st.sidebar.slider("Initial predator population", 2.0, 50.0, 5.0, 1.0)

st.sidebar.header("Simulation Settings")
t_end = st.sidebar.slider("Simulation duration", 10, 200, 100, 5)
noise = st.sidebar.slider("Observation noise level", 0.0, 0.20, 0.05, 0.01,
                          help="Noise standard deviation as a fraction of the mean population")

try:
    params = LotkaVolterraParams(alpha=alpha, beta=beta, delta=delta, gamma=gamma, x0=x0, y0=y0)
    clean_df = simulate(params, t_span=(0, t_end), t_eval_points=1000)
    noisy_df = generate_noisy_dataset(clean_df, noise_level=noise)
except Exception as e:
    st.error(f"Simulation failed: {e}")
    st.stop()

tab_sim, tab_ml, tab_real = st.tabs(["1. Simulation", "2. Machine learning", "3. Hudson Bay data"])

# ------------------------------------------------------------------ tab 1
with tab_sim:
    x_eq, y_eq = params.equilibrium
    drift = invariant_drift(params, clean_df)
    m1, m2, m3 = st.columns(3)
    m1.metric("Equilibrium (prey, predator)", f"({x_eq:.2f}, {y_eq:.2f})")
    m2.metric("Invariant drift (max relative)", f"{drift['max_rel']:.1e}",
              help="H = delta*x - gamma*ln(x) + beta*y - alpha*ln(y) is constant along exact solutions; its drift measures the numerical error.")
    m3.metric("Lowest prey value", f"{clean_df['prey'].min():.3g}")
    if clean_df["prey"].min() < 0.1 or clean_df["predator"].min() < 0.1:
        st.warning("A population falls below 0.1: this orbit is far from the equilibrium, the populations are almost extinct for long "
                   "periods and the noisy data are dominated by the lower clipping. Start closer to the equilibrium.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Temporal evolution")
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=clean_df["time"], y=clean_df["prey"], mode="lines", name="Prey (clean)", line=dict(color="#1f77b4", width=2)))
        fig_time.add_trace(go.Scatter(x=clean_df["time"], y=clean_df["predator"], mode="lines", name="Predator (clean)", line=dict(color="#d62728", width=2)))
        if noise > 0:
            fig_time.add_trace(go.Scatter(x=noisy_df["time"], y=noisy_df["prey_noisy"], mode="markers", name="Prey (noisy)", marker=dict(color="#1f77b4", size=3, opacity=0.4)))
            fig_time.add_trace(go.Scatter(x=noisy_df["time"], y=noisy_df["predator_noisy"], mode="markers", name="Predator (noisy)", marker=dict(color="#d62728", size=3, opacity=0.4)))
        fig_time.update_layout(xaxis_title="Time", yaxis_title="Population", template="plotly_white", height=450,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_time)

    with col2:
        st.subheader("Phase portrait (prey vs predator)")
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(x=clean_df["prey"], y=clean_df["predator"], mode="lines", name="Trajectory", line=dict(color="#2ca02c", width=2)))
        fig_phase.add_trace(go.Scatter(x=[clean_df["prey"].iloc[0]], y=[clean_df["predator"].iloc[0]], mode="markers", name="Initial state",
                                       marker=dict(color="black", size=10, symbol="circle-open", line=dict(width=2))))
        fig_phase.add_trace(go.Scatter(x=[x_eq], y=[y_eq], mode="markers", name="Equilibrium", marker=dict(color="#ff7f0e", size=10, symbol="x")))
        fig_phase.update_layout(xaxis_title="Prey population", yaxis_title="Predator population", template="plotly_white",
                                height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_phase)

    st.subheader("Generated dataset (for the ML tab)")
    st.dataframe(noisy_df.head(10))
    st.download_button("Download dataset as CSV", data=noisy_df.to_csv(index=False).encode("utf-8"),
                       file_name="lotka_volterra_dataset.csv", mime="text/csv")
    st.caption("Method: scipy.integrate.solve_ivp, RK45, rtol=atol=1e-8. Noise: Gaussian, standard deviation = noise level x mean population, "
               "values clipped below at 0.1.")


# ------------------------------------------------------------------ tab 2
@st.cache_data(show_spinner=False)
def run_ml(alpha, beta, delta, gamma, x0, y0, t_end, noise):
    p = LotkaVolterraParams(alpha=alpha, beta=beta, delta=delta, gamma=gamma, x0=x0, y0=y0)
    data = generate_noisy_dataset(simulate(p, t_span=(0, t_end), t_eval_points=1000), noise_level=noise)
    models, results, (X_test, y_test) = train_and_evaluate(data, verbose=False)
    unseen = unseen_orbit_evaluation(models, p, noise_level=noise, t_span=(0, t_end), verbose=False)
    preds = {name: m.predict(X_test) for name, m in models.items()}
    return results_to_frame(results), unseen, y_test, preds


with tab_ml:
    st.subheader("One-step-ahead prediction")
    st.markdown("Task: predict the state at t+1 from the state at t. The **persistence** baseline (\"next state = current state\") is "
                "included because with a small time step it is already very accurate; a model is only useful if it beats it.")
    ml_table, unseen, y_test, preds = run_ml(alpha, beta, delta, gamma, x0, y0, t_end, noise)

    st.markdown("**A. Same orbit** - chronological 80/20 split of one trajectory")
    st.dataframe(ml_table.round(4))

    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(y=y_test[:, 0], mode="lines", name="Prey (observed)", line=dict(color="black", width=1.5)))
    for name, style in [("LinearRegression", dict(color="#1f77b4", dash="dash")), ("RandomForestRegressor", dict(color="#d62728", dash="dot"))]:
        fig_ml.add_trace(go.Scatter(y=preds[name][:, 0], mode="lines", name=f"{name} (prey)", line=style))
    fig_ml.update_layout(template="plotly_white", height=380, xaxis_title="Test sample index", yaxis_title="Prey population",
                         margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_ml)

    st.markdown("**B. Unseen orbits** - same system, other initial conditions, models NOT retrained. "
                "This measures whether a model learned the dynamics or memorised one orbit.")
    for label, table in unseen.items():
        st.markdown(f"*{label}*")
        st.dataframe(table.round(4))
    st.caption("A ratio above 1 in the last two columns means the model is worse than simply repeating the current state.")

# ------------------------------------------------------------------ tab 3
@st.cache_data(show_spinner=False)
def run_real_fit():
    df = load_real_data()
    fit = fit_lotka_volterra(df, n_starts=25)
    ho = holdout_check(df, n_starts=15)
    return df, fit, ho


with tab_real:
    st.subheader("Fit to the Hudson Bay lynx-hare series (1900-1920)")
    st.markdown("Pelts traded (thousands) are a **proxy** for abundance. The model is fitted in these units, so alpha and gamma are rates per year, "
                "while beta and delta absorb the pelts-to-population scale and are not biological rates. "
                "The optimiser is restarted from several random points because the loss has many local minima (mostly wrong cycle periods).")
    with st.spinner("Fitting the model (multi-start optimisation)..."):
        real_df, fit, ho = run_real_fit()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("alpha (per year)", f"{fit.params['alpha']:.3f}")
    c2.metric("gamma (per year)", f"{fit.params['gamma']:.3f}")
    c3.metric("Fitted cycle period", f"{fit.period:.1f} years")
    c4.metric("R2 hare / lynx", f"{fit.metrics['hare']['r2']:.2f} / {fit.metrics['lynx']['r2']:.2f}")

    years = real_df["year"].to_numpy()
    t = years - years[0]
    t_fine = np.linspace(t[0], t[-1], 400)
    fine = simulate_fit(fit.theta, t_fine)
    fig_real = go.Figure()
    fig_real.add_trace(go.Scatter(x=years, y=real_df["hare"], mode="markers", name="Hare (observed)", marker=dict(color="#1f77b4")))
    fig_real.add_trace(go.Scatter(x=years, y=real_df["lynx"], mode="markers", name="Lynx (observed)", marker=dict(color="#d62728", symbol="square")))
    fig_real.add_trace(go.Scatter(x=years[0] + t_fine, y=fine[0], mode="lines", name="Hare (fitted)", line=dict(color="#1f77b4")))
    fig_real.add_trace(go.Scatter(x=years[0] + t_fine, y=fine[1], mode="lines", name="Lynx (fitted)", line=dict(color="#d62728")))
    fig_real.update_layout(template="plotly_white", height=420, xaxis_title="Year", yaxis_title="Pelts traded (thousands)",
                           margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_real)

    st.markdown(f"**Hold-out check:** refitting on {years[0]}-{years[ho['n_train'] - 1]} only, the forecast for the following years has an RMSE of "
                f"{ho['hare']['rmse']:.1f} (hare) and {ho['lynx']['rmse']:.1f} (lynx) thousand pelts, versus "
                f"{ho['hare']['rmse_mean_baseline']:.1f} and {ho['lynx']['rmse_mean_baseline']:.1f} for a constant forecast at the training mean.")
    st.caption("Limitations: 21 points per species, one region, no carrying capacity, no seasonality.")
