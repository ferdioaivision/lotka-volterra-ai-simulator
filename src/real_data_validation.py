"""
Real Data Validation - Hudson Bay Lynx-Hare Dataset
Scientific validation of Lotka-Volterra model against historical data (1900-1920)

This module transforms the project from simulation to case study.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import os

def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def simulate_with_params(params, t_span, y0, t_eval):
    alpha, beta, delta, gamma = params
    sol = solve_ivp(
        lambda t, z: lotka_volterra(t, z, alpha, beta, delta, gamma),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8
    )
    return sol.y[0], sol.y[1]

def loss_function(params, real_df, y0, t_eval):
    """MSE between scaled simulation and real data"""
    try:
        # Prevent negative or zero params
        if any(p <= 0 for p in params):
            return 1e10
        x_sim, y_sim = simulate_with_params(params, (0, 20), y0, t_eval)
        # Scale simulation to real data magnitude (real data in thousands)
        # Normalize both to [0,1] for comparison of dynamics, not absolute values
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        real_hare_norm = scaler.fit_transform(real_df[['hare']]).flatten()
        # Simple loss on normalized shapes - focus on cycle matching
        # We compare hare vs prey, lynx vs predator
        x_norm = (x_sim - x_sim.min()) / (x_sim.max() - x_sim.min() + 1e-8)
        y_norm = (y_sim - y_sim.min()) / (y_sim.max() - y_sim.min() + 1e-8)
        loss_hare = np.mean((x_norm - real_hare_norm)**2)
        real_lynx_norm = scaler.fit_transform(real_df[['lynx']]).flatten()
        loss_lynx = np.mean((y_norm - real_lynx_norm)**2)
        return loss_hare + loss_lynx
    except:
        return 1e10

def fit_to_real_data():
    # Load real data
    data_path = os.path.join(os.path.dirname(__file__), '../data/lynx_hare_real.csv')
    if not os.path.exists(data_path):
        # Fallback inline
        real_df = pd.DataFrame({
            'year': list(range(1900, 1921)),
            'hare': [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7],
            'lynx': [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
        })
    else:
        real_df = pd.read_csv(data_path)
    
    t_eval = np.linspace(0, 20, len(real_df))
    y0 = [real_df['hare'].iloc[0], real_df['lynx'].iloc[0]]
    
    # Initial guess from literature
    x0_params = [0.5, 0.02, 0.02, 0.5]
    
    print("Fitting Lotka-Volterra parameters to Hudson Bay data...")
    print(f"Initial guess: alpha={x0_params[0]}, beta={x0_params[1]}, delta={x0_params[2]}, gamma={x0_params[3]}")
    
    result = minimize(
        loss_function, x0_params,
        args=(real_df, y0, t_eval),
        method='Nelder-Mead',
        options={'maxiter': 500, 'disp': True}
    )
    
    fitted_params = result.x
    print(f"\nFitted params: alpha={fitted_params[0]:.4f}, beta={fitted_params[1]:.4f}, delta={fitted_params[2]:.4f}, gamma={fitted_params[3]:.4f}")
    print(f"Final loss: {result.fun:.6f}")
    
    # Simulate with fitted params
    x_fit, y_fit = simulate_with_params(fitted_params, (0, 20), y0, t_eval)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(real_df['year'], real_df['hare'], 'o-', label='Hare (Real)', color='#1f77b4', linewidth=2)
    axes[0].plot(real_df['year'], real_df['lynx'], 's-', label='Lynx (Real)', color='#d62728', linewidth=2)
    axes[0].set_title('Hudson Bay - Real Data (1900-1920)')
    axes[0].set_ylabel('Pelts (x1000)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(real_df['year'], x_fit, '--', label=f'Prey Simulated (fitted)', color='#1f77b4', linewidth=2)
    axes[1].plot(real_df['year'], y_fit, '--', label=f'Predator Simulated (fitted)', color='#d62728', linewidth=2)
    axes[1].set_title(f'Simulation with Fitted Params: alpha={fitted_params[0]:.2f}, beta={fitted_params[1]:.3f}, delta={fitted_params[2]:.3f}, gamma={fitted_params[3]:.2f}')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Simulated Population')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '../figures/real_vs_fitted.png'), dpi=300)
    plt.show()
    
    return fitted_params, real_df

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), '../figures'), exist_ok=True)
    fit_to_real_data()
