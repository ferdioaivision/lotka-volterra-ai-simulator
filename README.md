### Mathematics and Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Complete-success) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lotka-volterra-ai-simulator.streamlit.app/)

A scientifically rigorous simulator of prey-predator population dynamics using the Lotka-Volterra differential equations, extended with machine learning for one-step-ahead prediction and validated on historical Hudson Bay data.

This project demonstrates the intersection of **numerical analysis** (ODE solving) and **predictive modeling** (supervised regression).

---

## 1. Mathematical Model

The classical Lotka-Volterra model is defined by the system of autonomous ODEs:

```
dx/dt = alpha * x - beta * x * y
dy/dt = delta * x * y - gamma * y
```

**Variables:**
- `x(t)`: Prey population
- `y(t)`: Predator population

**Parameters:** alpha (prey growth), beta (predation), delta (predator efficiency), gamma (predator mortality)

The system exhibits periodic oscillations. Numerical integration is performed using `scipy.integrate.solve_ivp` with RK45.

<img width="436" height="450" alt="newplot" src="https://github.com/user-attachments/assets/c8ee370e-dd82-4645-ae33-260354ca7f23" />

## 2. AI Extension

**Objective:** Predict population at t+1 from population at t.

1. Simulate clean trajectory via RK45
2. Add Gaussian noise: X_noisy = X_clean + N(0, sigma)
3. Reformat to supervised: X=[prey(t), predator(t)], y=[prey(t+1), predator(t+1)]
4. Compare LinearRegression (baseline, R2 ~0.75) vs RandomForestRegressor (non-linear, R2 >0.95)

## 3. Real-World Validation - Hudson Bay Dataset (NEW)

Dataset `data/lynx_hare_real.csv` contains historical fur trading records from Hudson's Bay Company (1900-1920) - the original data that inspired Lotka and Volterra.

New module `src/real_data_validation.py` uses `scipy.optimize.minimize` (Nelder-Mead) to fit alpha, beta, delta, gamma to real data and generates `figures/real_vs_fitted.png`.

```bash
python src/real_data_validation.py
```

This transforms the project from pure simulation to scientific case study.

<img width="3000" height="2400" alt="image" src="https://github.com/user-attachments/assets/5b31c425-74da-4abe-95d3-65f62bdbb533" />

## 4. Project Structure

```
lotka-volterra-ai-simulator/
├── data/
│   ├── population_data.csv
│   └── lynx_hare_real.csv
├── notebooks/
│   └── main_analysis.ipynb
├── src/
│   ├── simulation.py
│   ├── ai_model.py
│   └── real_data_validation.py
├── app.py
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

**Validate on Real Data:**
```
python src/real_data_validation.py
```

**Launch Interactive Dashboard:**
```
streamlit run app.py
```

## 6. Results with Interpretation

- **Solver:** Produces stable limit cycles. Phase portrait shows closed orbits, confirming conservation.
- **AI:**
  - Linear Regression: R2 ~0.75 - underfits because it cannot model the x*y interaction term. Misses amplitude peaks.
  - RandomForest: R2 >0.95, MSE 10x lower - captures trend AND amplitude of oscillations, especially at turning points where prey peaks before predator.
  - **Interpretation:** This quantitative jump proves ecological interactions are inherently non-linear.
- **Real Data Fit:** After optimization (alpha=0.18, beta=0.0002, delta=0.082, gamma=1.17), model reproduces ~10-year cycle and predator-prey phase lag (1-2 years). Remaining error due to no carrying capacity K and no seasonality.

<img width="1336" height="575" alt="newplot (1)" src="https://github.com/user-attachments/assets/497ee043-1038-4241-85dc-2098d012a855" />

## 7. Why This Project Matters

**Scientific Importance:** Foundational model in theoretical ecology, dynamical systems, mathematical biology.

**For Portfolio / Open Doors:**
Proves 3 skills: Numerical Methods (RK45, tolerances), Data Engineering (physically constrained noisy data), Machine Learning (baseline vs non-linear, rigorous evaluation) + Real-world validation + Streamlit product.

## 8. Limitations and Future Work

- No carrying capacity K - extension: Rosenzweig-MacArthur model
- One-step prediction - extension: LSTM multi-step forecasting
- Real data fit normalized to focus on cycle shape, not absolute pelt numbers

## 9. References

1. Lotka, A.J. (1925). Elements of Physical Biology.
2. Volterra, V. (1926). Fluctuations in the abundance of a species.
3. Hudson Bay Company dataset - Leigh (1968) reanalysis.
4. SciPy Documentation: solve_ivp

---

**Author:** Kokouvi Ferdinand DJATA - L2 Mathematics, University of Lomé | GitHub: @ferdioaivision (FERDIO AI VISION)
