"""
Population Dynamics Simulator - Lotka-Volterra Model
----------------------------------------------------
Numerical implementation of the classical Lotka-Volterra predator-prey model.

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

Conserved quantity (first integral) of the classical model:
    H(x, y) = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)
Every orbit is a closed level curve of H (a neutrally stable centre, not a
limit cycle). Since RK45 does not preserve H exactly, its drift along a
numerical trajectory is a direct check of the integration accuracy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Paths are resolved from this file, so the scripts work from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Lower bound applied to noisy observations (populations must stay positive).
MIN_POPULATION = 0.1


@dataclass(frozen=True)
class LotkaVolterraParams:
    """Immutable container for Lotka-Volterra model parameters.

    The default initial state (10, 5) is close to the equilibrium
    (gamma/delta, alpha/beta) = (4, 2.75), so populations stay at biologically
    meaningful levels (no value far below one individual).
    """
    alpha: float = 1.1   # Prey growth rate
    beta: float = 0.4    # Predation rate
    delta: float = 0.1   # Predator growth efficiency
    gamma: float = 0.4   # Predator mortality rate
    x0: float = 10.0     # Initial prey population
    y0: float = 5.0      # Initial predator population

    def __post_init__(self):
        if any(v <= 0 for v in [self.alpha, self.beta, self.delta, self.gamma, self.x0, self.y0]):
            raise ValueError("All parameters must be strictly positive.")

    @property
    def equilibrium(self) -> Tuple[float, float]:
        """Non-trivial equilibrium (prey, predator) = (gamma/delta, alpha/beta)."""
        return self.gamma / self.delta, self.alpha / self.beta


def lotka_volterra_ode(t: float, z: np.ndarray, alpha: float, beta: float, delta: float, gamma: float) -> np.ndarray:
    """
    Right-hand side of the Lotka-Volterra system.

    Args:
        t: Time (required by solve_ivp, unused because the system is autonomous)
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
        method: Integration method (RK45 = Dormand-Prince 5(4))
        rtol, atol: Relative and absolute tolerances

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

    return pd.DataFrame({
        "time": solution.t,
        "prey": solution.y[0],
        "predator": solution.y[1]
    })


def invariant(params: LotkaVolterraParams, prey: np.ndarray, predator: np.ndarray) -> np.ndarray:
    """First integral H(x, y) of the classical Lotka-Volterra system (constant along exact orbits)."""
    return (params.delta * prey - params.gamma * np.log(prey)
            + params.beta * predator - params.alpha * np.log(predator))


def invariant_drift(params: LotkaVolterraParams, df: pd.DataFrame) -> Dict[str, float]:
    """
    Drift of H along a numerical trajectory, used to check the solver.

    Returns:
        {'h0': H at t=0, 'max_abs': max |H - H0|, 'max_rel': max |H - H0| / |H0|}
    """
    h = invariant(params, df["prey"].to_numpy(), df["predator"].to_numpy())
    dev = np.abs(h - h[0])
    return {"h0": float(h[0]), "max_abs": float(dev.max()), "max_rel": float(dev.max() / abs(h[0]))}


def generate_noisy_dataset(
    clean_df: pd.DataFrame,
    noise_level: float = 0.05,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Adds Gaussian observation noise to a clean trajectory.

    The noise standard deviation is `noise_level` times the MEAN of each
    population (not proportional to the instantaneous value). Noisy values are
    then clipped below at MIN_POPULATION, because a population cannot be
    negative; the clipping is not a rescaling, it only truncates the tail.

    Args:
        clean_df: DataFrame from simulate()
        noise_level: Noise standard deviation as a fraction of the mean population
        random_seed: For reproducibility

    Returns:
        DataFrame with additional columns ['prey_noisy', 'predator_noisy']
    """
    rng = np.random.default_rng(random_seed)
    df = clean_df.copy()

    prey_std = noise_level * df["prey"].mean()
    predator_std = noise_level * df["predator"].mean()

    df["prey_noisy"] = (df["prey"] + rng.normal(0, prey_std, size=len(df))).clip(lower=MIN_POPULATION)
    df["predator_noisy"] = (df["predator"] + rng.normal(0, predator_std, size=len(df))).clip(lower=MIN_POPULATION)
    return df


def clipped_fraction(noisy_df: pd.DataFrame) -> float:
    """Fraction of noisy observations that were truncated at MIN_POPULATION."""
    cols = ["prey_noisy", "predator_noisy"]
    return float((noisy_df[cols] <= MIN_POPULATION).to_numpy().mean())


if __name__ == "__main__":
    params = LotkaVolterraParams()
    print(f"Simulating with parameters: {params}")
    print(f"Equilibrium (prey, predator): {params.equilibrium}")

    clean_data = simulate(params, t_span=(0, 100), t_eval_points=1000)
    noisy_data = generate_noisy_dataset(clean_data, noise_level=0.05)
    drift = invariant_drift(params, clean_data)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "population_data.csv"
    noisy_data.to_csv(output_path, index=False, float_format="%.6f")

    print(f"Data shape: {noisy_data.shape}")
    print(f"Prey range: {clean_data['prey'].min():.3f} - {clean_data['prey'].max():.3f}")
    print(f"Predator range: {clean_data['predator'].min():.3f} - {clean_data['predator'].max():.3f}")
    print(f"Invariant drift: max |H-H0| = {drift['max_abs']:.2e} (relative {drift['max_rel']:.2e})")
    print(f"Clipped noisy observations: {100 * clipped_fraction(noisy_data):.2f}%")
    print(f"Data saved to: {output_path}")
