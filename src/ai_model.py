"""
AI Model Module - Predictive Modeling
----------------------------------------------
Compares a baseline linear model vs a non-linear model for one-step-ahead
prediction of Lotka-Volterra dynamics.

Dataset format expected: columns ['time', 'prey_noisy', 'predator_noisy']
Task: X(t) = [prey(t), predator(t)] -> y(t) = [prey(t+1), predator(t+1)]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelEvaluation:
    model_name: str
    mse_prey: float
    mse_predator: float
    r2_prey: float
    r2_predator: float


def create_supervised_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts time series into supervised learning format for one-step prediction.

    Args:
        df: DataFrame with prey_noisy and predator_noisy

    Returns:
        X: shape (n-1, 2) - current state
        y: shape (n-1, 2) - next state
    """
    data = df[["prey_noisy", "predator_noisy"]].values
    X = data[:-1]  # t
    y = data[1:]   # t+1
    return X, y


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Trains LinearRegression and RandomForestRegressor and evaluates performance.
    This is the core AI extension for L2 level.
    """
    X, y = create_supervised_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    models = {
        "LinearRegression (Baseline)": LinearRegression(),
        "RandomForestRegressor (Non-linear)": RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        eval_result = ModelEvaluation(
            model_name=name,
            mse_prey=mean_squared_error(y_test[:, 0], y_pred[:, 0]),
            mse_predator=mean_squared_error(y_test[:, 1], y_pred[:, 1]),
            r2_prey=r2_score(y_test[:, 0], y_pred[:, 0]),
            r2_predator=r2_score(y_test[:, 1], y_pred[:, 1])
        )
        results.append(eval_result)
        trained_models[name] = model

        print(f"\n--- {name} ---")
        print(f"R2 Prey: {eval_result.r2_prey:.4f} | R2 Predator: {eval_result.r2_predator:.4f}")
        print(f"MSE Prey: {eval_result.mse_prey:.4f} | MSE Predator: {eval_result.mse_predator:.4f}")

    return trained_models, results, (X_test, y_test)


if __name__ == "__main__":
    # Example usage - expects data/population_data.csv to exist
    try:
        df = pd.read_csv("../data/population_data.csv")
        train_and_evaluate(df)
    except FileNotFoundError:
        print("Data file not found. Please run src/simulation.py first to generate data.")
