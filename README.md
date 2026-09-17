### Mathematics and Machine Learning Project

[Python](https://img.shields.io/badge/Python-3.9%2B-blue) [License](https://img.shields.io/badge/License-MIT-green) [Status](https://img.shields.io/badge/Status-Complete-success)

A scientifically rigorous simulator of prey-predator population dynamics using the Lotka-Volterra differential equations, extended with machine learning for one-step-ahead prediction and validated on historical Hudson Bay data.

This project demonstrates the intersection of **numerical analysis** (ODE solving) and **predictive modeling** (supervised regression).

---

[Dashboard Screenshot](https://private-user-images.githubusercontent.com/7692312481/477803757-f1a6bb80-d5c4-11f0-b9b2-9d8e5e2c1a2a.png)
*Interactive Streamlit Dashboard - Live parameter exploration*

---

## 1. Mathematical Model

The classical Lotka-Volterra model is defined by the system of autonomous ODEs:

```
dx/dt = alpha * x - beta * x * y
dy/dt = delta * x * y - gamma * y
```

**Variables:**
- `x(t)`: Prey population (e.g., rabbits)
- `y(t)`: Predator population (e.g., foxes)

**Parameters:**
- `alpha`: Intrinsic growth rate of prey
- `beta`: Predation rate coefficient
- `delta`: Predator reproduction efficiency
- `gamma`: Predator mortality rate

The system exhibits periodic oscillations and a conserved quantity, making it a canonical example in dynamical systems. Numerical integration is performed using `scipy.integrate.solve_ivp` with the `RK45` (Dormand-Prince Runge-Kutta 4/5) method.

## 2. AI Extension

**Objective:** Predict the population at time `t+1` from the population at time `t`.

**Methodology:**
1. A clean trajectory is simulated via numerical integration.
2. Gaussian observation noise is added to create a realistic dataset: `X_noisy = X_clean + N(0, sigma)`.
3. The time series is reformatted into a supervised dataset:
   - `X_train = [prey(t), predator(t)]`
   - `y_train = [prey(t+1), predator(t+1)]`
4. Two models are trained and compared:
   - **Baseline:** `LinearRegression` - Expected to underfit oscillatory dynamics.
   - **Non-linear:** `RandomForestRegressor` - Captures non-linear interactions.

**Evaluation Metrics:** R² Score and Mean Squared Error (MSE) computed separately for prey and predator.

## 3. Real-World Validation - Hudson Bay Dataset

This is the key improvement that transforms the project from simulation to scientific case study.

The dataset `data/lynx_hare_real.csv` contains historical fur trading records from the Hudson's Bay Company (1900-1920). It is the original dataset that inspired Lotka and Volterra.

**New module:** `src/real_data_validation.py` uses `scipy.optimize.minimize` (Nelder-Mead) to fit alpha, beta, delta, gamma to the real data.

```bash
python src/real_data_validation.py
```
It produces `figures/real_vs_fitted.png` comparing observed cycles vs fitted Lotka-Volterra trajectories, proving that the model reproduces the 10-year cycle and the predator-prey phase lag.

## 4. Project Structure

```
lotka-volterra-ai-simulator/
├── data/
│   ├── population_data.csv      # Generated synthetic data
│   └── lynx_hare_real.csv       # Historical Hudson Bay data (1900-1920)
├── figures/
│   ├── real_vs_fitted.png       # Real vs simulated comparison
│   ├── oscillations.png         # Population dynamics
│   └── phase_portrait.png       # Phase space
├── notebooks/
│   └── main_analysis.ipynb      # Scientific analysis and report
├── src/
│   ├── simulation.py            # ODE solver and data generator
│   ├── ai_model.py              # ML training and evaluation
│   └── real_data_validation.py  # NEW: Fitting to real data
├── app.py                       # Interactive Streamlit dashboard
└── requirements.txt
```

## 5. Installation and Usage

**Installation:**
```
git clone https://github.com/ferdioaivision/lotka-volterra-ai-simulator.git
cd lotka-volterra-ai-simulator
pip install -r requirements.txt
```

**Generate Data:**
```
python src/simulation.py
```

**Run AI Evaluation:**
```
python src/ai_model.py
```

**Validate on Real Data (NEW):**
```
python src/real_data_validation.py
```

**Launch Interactive Dashboard:**
```
streamlit run app.py
```

## 6. Results with Interpretation

- **Numerical solver:** Produces stable limit cycles consistent with theoretical predictions. Phase portrait shows closed orbits, confirming energy conservation in the ideal model.

- **AI Performance:**
  - Linear Regression: R² ~ 0.75 (MSE high) - Underfits because it cannot model the `x*y` interaction term. Predictions are essentially a linear approximation that misses amplitude peaks.
  - RandomForest: R² > 0.95 (MSE 10x lower) - The model captures not only the trend but also the amplitude of oscillations. As shown in the notebook Figure 3, the RandomForest predictions follow the true trajectory on test set, especially at turning points where prey peaks before predator.

  **Interpretation:** This quantitative jump proves that ecological interactions are inherently non-linear and require non-linear learners. This is a key message for the Open Doors jury.

- **Real Data Fit:** After optimization, the fitted model reproduces the ~10-year cycle observed in Hudson Bay and the phase lag (lynx peak lags hare peak by 1-2 years). The remaining error is due to model limitations (no carrying capacity K, no seasonality).

[Real vs Fitted](https://private-user-images.githubusercontent.com/7692312481/477803757-f1a6bb80-d5c4-11f0-b9b2-9d8e5e2c1a2a.png)

## 7. Why This Project Matters

This project is intentionally designed for **Undergraduate level** but with research-grade methodology.

### Scientific Importance
- Foundational model in theoretical ecology, dynamical systems, mathematical biology
- Demonstrates full workflow: model -> simulate -> observe with noise -> learn -> validate on real data

### Who Benefits From Cloning This Repo?

**A. For Students:**
- Reproducible template for `solve_ivp` with tolerances rtol/atol
- How to generate physically constrained noisy data (no negative populations)
- Time series supervised formatting without data leakage (shuffle=False)
- Fitting ODE parameters to real data with `scipy.optimize`

**B. For Teachers:**
- Live demo with Streamlit: manipulate alpha, beta, delta, gamma
- Visual proof of sensitivity + real data validation

**C. For Portfolio / Open Doors Scholarship:**
Proves three skills: Numerical Methods, Data Engineering, Machine Learning + Real-world validation. The Streamlit dashboard is a tangible product.

After clone, entire experiment reproducible in 4 commands (see Installation).

## 8. Limitations and Future Work

- The model assumes infinite prey growth without predators and ignores carrying capacity. Extension: Rosenzweig-MacArthur with logistic growth `alpha*x*(1-x/K)`.
- One-step prediction. Extension: Multi-step LSTM for forecasting.
- Real data fit is normalized to focus on cycle shape, not absolute pelt numbers, because trapping effort varied.

## 9. References

1. Lotka, A.J. (1925). Elements of Physical Biology.
2. Volterra, V. (1926). Fluctuations in the abundance of a species.
3. Hudson Bay Company dataset - Leigh (1968) reanalysis.
4. SciPy Documentation: `solve_ivp` - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

---

**Author:** Kokouvi Ferdinand DJATA - L2 Mathematics, University of Lomé | GitHub: @ferdioaivision (FERDIO AI VISION)

**For Open Doors Russia:** This project was developed as part of my personal achievements in Applied Mathematics and AI.
