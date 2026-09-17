"""
Population Dynamics Simulator - Lotka-Volterra Model
----------------------------------------------------
Scientific implementation of the classical Lotka-Volterra predator-prey model.

Author: Kokouvi Ferdinand DJATA
Model:
    dx/dt = alpha * x - beta * x * y
    dy/dt = delta * x * y - gamma * y

Where:
    x: prey population
    y: predator population
    alpha: prey intrinsic growth rate
    beta: predation rate coefficient
    delta: predator reproduction rate per prey consumed
    gamma: predator mortality rate
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class LotkaVolterraParams:
    """Immutable container for Lotka-Volterra model parameters."""
    alpha: float = 1.1   # Prey growth rate
    beta: float = 0.4    # Predation rate
    delta: float = 0.1   # Predator growth efficiency
    gamma: float = 0.4   # Predator mortality rate
    x0: float = 40.0     # Initial prey population
    y0: float = 9.0      # Initial predator population

    def __post_init__(self):
        if any(v <= 0 for v in [self.alpha, self.beta, self.delta, self.gamma, self.x0, self.y0]):
            raise ValueError("All parameters must be strictly positive.")


def lotka_volterra_ode(t: float, z: np.ndarray, alpha: float, beta: float, delta: float, gamma: float) -> np.ndarray:
    """
    Defines the system of ordinary differential equations for Lotka-Volterra dynamics.

    Args:
        t: Time variable (required by solve_ivp, not used explicitly as system is autonomous)
        z: State vector [x, y] where x=prey, y=predator
        alpha, beta, delta, gamma: Model parameters

    Returns:
        dz/dt vector [dx/dt, dy/dt]
    """
    x, y = z
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return np.array([dx_dt, dy_dt])


def simulate(
    params: LotkaVolterraParams,
    t_span: Tuple[float, float] = (0, 100),
    t_eval_points: int = 1000,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-8
) -> pd.DataFrame:
    """
    Simulates the Lotka-Volterra model using scipy.integrate.solve_ivp.

    Args:
        params: Model parameters container
        t_span: Simulation time interval (t_start, t_end)
        t_eval_points: Number of time points to evaluate
        method: Integration method (RK45 is standard Runge-Kutta 4-5)
        rtol, atol: Relative and absolute tolerances for solver accuracy

    Returns:
        DataFrame with columns ['time', 'prey', 'predator']
    """
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)
    z0 = np.array([params.x0, params.y0])

    solution = solve_ivp(
        fun=lambda t, z: lotka_volterra_ode(t, z, params.alpha, params.beta, params.delta, params.gamma),
        t_span=t_span,
        y0=z0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    df = pd.DataFrame({
        "time": solution.t,
        "prey": solution.y[0],
        "predator": solution.y[1]
    })
    return df


def generate_noisy_dataset(
    clean_df: pd.DataFrame,
    noise_level: float = 0.05,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic dataset by adding Gaussian noise to simulate observation error.
    Essential for training robust AI models.

    Args:
        clean_df: DataFrame from simulate()
        noise_level: Standard deviation of noise as fraction of mean population (e.g., 0.05 = 5%)
        random_seed: For reproducibility

    Returns:
        DataFrame with additional columns ['prey_noisy', 'predator_noisy']
    """
    rng = np.random.default_rng(random_seed)
    df = clean_df.copy()

    # Noise proportional to signal amplitude to avoid negative populations
    prey_std = noise_level * df["prey"].mean()
    predator_std = noise_level * df["predator"].mean()

    df["prey_noisy"] = df["prey"] + rng.normal(0, prey_std, size=len(df))
    df["predator_noisy"] = df["predator"] + rng.normal(0, predator_std, size=len(df))

    # Physical constraint: populations cannot be negative
    df["prey_noisy"] = df["prey_noisy"].clip(lower=0.1)
    df["predator_noisy"] = df["predator_noisy"].clip(lower=0.1)

    return df


if __name__ == "__main__":
    # Professional execution entry point for data generation
    params = LotkaVolterraParams(
        alpha=1.1,
        beta=0.4,
        delta=0.1,
        gamma=0.4,
        x0=40.0,
        y0=9.0
    )

    print(f"Simulating with parameters: {params}")
    clean_data = simulate(params, t_span=(0, 100), t_eval_points=1000)
    noisy_data = generate_noisy_dataset(clean_data, noise_level=0.05)

    output_path = "../data/population_data.csv"
    # Ensure we save from src/ context
    import os
    os.makedirs("../data", exist_ok=True)
    noisy_data.to_csv(output_path, index=False, float_format="%.6f")

    print(f"Simulation successful. Data shape: {noisy_data.shape}")
    print(f"Data saved to: {output_path}")
    print(noisy_data.head().to_string(index=False))
