"""
AI Model Module - One-step-ahead prediction
-------------------------------------------
Task: X(t) = [prey(t), predator(t)]  ->  y(t) = [prey(t+1), predator(t+1)].

Three predictors are compared:
  * Persistence   : predicts y(t+1) = X(t). With a small time step consecutive
                    states are almost identical, so this "do nothing" baseline
                    already scores a very high R2. Any model must be judged
                    against it, not against zero.
  * LinearRegression
  * RandomForestRegressor

Two evaluations are provided:
  1. Same-orbit test: chronological 80/20 split of ONE trajectory. The test
     window lies on the same closed orbit as the training data.
  2. Unseen-orbit test: the models trained on one orbit are applied to other
     orbits (different initial conditions, same model parameters). This is the
     evaluation that measures whether a model learned the dynamics or only
     memorised one orbit.

Dataset format: columns ['time', 'prey_noisy', 'predator_noisy'].
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:  # imported as part of the "src" package (Streamlit app)
    from .simulation import (DATA_DIR, LotkaVolterraParams, generate_noisy_dataset,
                             simulate)
except ImportError:  # run as a script / from the notebook with src on sys.path
    from simulation import (DATA_DIR, LotkaVolterraParams, generate_noisy_dataset,
                            simulate)

PERSISTENCE = "Persistence (baseline)"
LINEAR = "LinearRegression"
FOREST = "RandomForestRegressor"


@dataclass
class ModelEvaluation:
    model_name: str
    mse_prey: float
    mse_predator: float
    r2_prey: float
    r2_predator: float


class PersistenceBaseline:
    """Predicts that the next state equals the current state."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array(X, copy=True)


def create_supervised_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a time series into supervised format for one-step prediction.

    Returns:
        X: shape (n-1, 2) - state at t
        y: shape (n-1, 2) - state at t+1
    """
    data = df[["prey_noisy", "predator_noisy"]].to_numpy()
    return data[:-1], data[1:]


def build_models(random_state: int = 42) -> Dict[str, object]:
    return {
        PERSISTENCE: PersistenceBaseline(),
        LINEAR: LinearRegression(),
        FOREST: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
    }


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelEvaluation:
    return ModelEvaluation(
        model_name=name,
        mse_prey=mean_squared_error(y_true[:, 0], y_pred[:, 0]),
        mse_predator=mean_squared_error(y_true[:, 1], y_pred[:, 1]),
        r2_prey=r2_score(y_true[:, 0], y_pred[:, 0]),
        r2_predator=r2_score(y_true[:, 1], y_pred[:, 1]),
    )


def results_to_frame(results: List[ModelEvaluation]) -> pd.DataFrame:
    """Results table, with the MSE ratio relative to the persistence baseline (<1 = better than baseline)."""
    df = pd.DataFrame([r.__dict__ for r in results]).set_index("model_name")
    base = df.loc[PERSISTENCE]
    df["mse_prey_vs_persistence"] = df["mse_prey"] / base["mse_prey"]
    df["mse_predator_vs_persistence"] = df["mse_predator"] / base["mse_predator"]
    return df


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, verbose: bool = True):
    """
    Trains the models on the first (1 - test_size) of the trajectory and evaluates
    them on the remaining part (chronological split, no shuffling).

    Returns:
        trained_models, results (list of ModelEvaluation), (X_test, y_test)
    """
    X, y = create_supervised_dataset(df)
    X_train, X_test, y_train, y_test = _chronological_split(X, y, test_size)

    trained, results = {}, []
    for name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, y_test, model.predict(X_test)))
        trained[name] = model

    if verbose:
        print("\nSame-orbit test (chronological 80/20 split)")
        print(results_to_frame(results).round(4).to_string())
    return trained, results, (X_test, y_test)


def _chronological_split(X, y, test_size):
    n_train = int(len(X) * (1 - test_size))
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def related_orbits(params: LotkaVolterraParams, inner: float = 0.4, outer: float = 1.8) -> Dict[str, LotkaVolterraParams]:
    """
    Two other orbits of the SAME system, obtained by scaling the initial
    distance to the equilibrium: a smaller orbit (inside the training orbit)
    and a larger one (outside it).
    """
    x_eq, y_eq = params.equilibrium

    def scaled(k: float) -> LotkaVolterraParams:
        return LotkaVolterraParams(
            alpha=params.alpha, beta=params.beta, delta=params.delta, gamma=params.gamma,
            x0=max(x_eq + k * (params.x0 - x_eq), 0.5),
            y0=max(y_eq + k * (params.y0 - y_eq), 0.5),
        )

    return {"inner orbit": scaled(inner), "outer orbit": scaled(outer)}


def unseen_orbit_evaluation(trained_models: Dict[str, object], params: LotkaVolterraParams,
                            noise_level: float = 0.05, t_span=(0, 100), t_eval_points: int = 1000,
                            seed: int = 7, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Applies already-trained models to orbits they never saw during training."""
    out = {}
    for label, orbit_params in related_orbits(params).items():
        noisy = generate_noisy_dataset(simulate(orbit_params, t_span, t_eval_points), noise_level, seed)
        X, y = create_supervised_dataset(noisy)
        results = [evaluate(name, y, model.predict(X)) for name, model in trained_models.items()]
        out[label] = results_to_frame(results)
        if verbose:
            print(f"\nUnseen {label} (x0={orbit_params.x0:.2f}, y0={orbit_params.y0:.2f})")
            print(out[label].round(4).to_string())
    return out


if __name__ == "__main__":
    csv_path = DATA_DIR / "population_data.csv"
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise SystemExit(f"{csv_path} not found. Run 'python src/simulation.py' first to generate it.")

    models, _, _ = train_and_evaluate(data)
    unseen_orbit_evaluation(models, LotkaVolterraParams())
