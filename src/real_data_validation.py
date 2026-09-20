"""
Real Data Validation - Hudson Bay Lynx-Hare Dataset (1900-1920)
---------------------------------------------------------------
Fits the Lotka-Volterra model to a classic historical series (pelts traded, in
thousands, used as a proxy for abundance) and checks the fit.

Method
  * The model is fitted directly in the units of the data (thousands of pelts).
    Scale factors between pelts and true population sizes are absorbed by
    beta and delta, so those two are NOT biological rates; alpha and gamma
    (per year) keep their meaning. The initial state (x0, y0) is fitted too,
    because the first observation is itself noisy.
  * Loss: sum of squared residuals, each species divided by its own standard
    deviation, so both species weigh equally while absolute amplitudes are kept.
  * The loss surface has many local minima (mostly wrong cycle periods), so the
    optimisation is restarted from many random points and the best result kept.
  * Hold-out check: the model is refitted on 1900-1912 only and asked to
    forecast 1913-1920.

Limitations: 21 points per species, one region, pelt counts are a proxy for
abundance, and the classical model has no carrying capacity or seasonality.
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "lynx_hare_real.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

PARAM_NAMES = ["alpha", "beta", "delta", "gamma", "x0", "y0"]
# Bounds on the parameters (natural scale), applied in log space.
_LOWER = np.log([0.05, 1e-4, 1e-4, 0.05, 2.0, 1.0])
_UPPER = np.log([3.0, 1.0, 1.0, 3.0, 300.0, 300.0])


def load_real_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Loads the dataset; there is deliberately no silent fallback to embedded data."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _solve(theta: np.ndarray, t_eval: np.ndarray, t_end: Optional[float] = None, rtol: float = 1e-8):
    alpha, beta, delta, gamma, x0, y0 = np.exp(theta)
    t_end = float(t_eval[-1]) if t_end is None else t_end
    return solve_ivp(
        lambda _, z: [alpha * z[0] - beta * z[0] * z[1], delta * z[0] * z[1] - gamma * z[1]],
        (float(t_eval[0]), t_end), [x0, y0], t_eval=t_eval, method="RK45", rtol=rtol, atol=1e-10,
    )


def simulate_fit(theta: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Returns an array of shape (2, len(t_eval)): simulated hare and lynx."""
    sol = _solve(theta, t_eval)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        raise RuntimeError("Integration failed for these parameters.")
    return sol.y


def estimate_period(theta: np.ndarray, horizon: float = 200.0) -> float:
    """Period of the fitted cycle, from the spacing of prey peaks on a long simulation (nan if < 2 peaks)."""
    t = np.linspace(0.0, horizon, 8001)
    sol = _solve(theta, t)
    if not sol.success or sol.y.shape[1] != len(t):
        return float("nan")
    peaks, _ = find_peaks(sol.y[0])
    return float(np.median(np.diff(t[peaks]))) if len(peaks) >= 2 else float("nan")


@dataclass
class FitResult:
    params: Dict[str, float]
    theta: np.ndarray
    cost: float                # mean squared residual on the fitted points (normalised if weighted)
    sse: float                 # sum of squared errors in data units (thousand pelts^2), both species
    n_starts: int
    n_near_best: int           # starts that ended within 5% of the best cost
    period: float              # years
    equilibrium: tuple         # (gamma/delta, alpha/beta) in thousands of pelts
    metrics: Dict[str, Dict[str, float]]  # per species: rmse, r2, corr on the fitted points


def _metrics(sim: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((sim - obs) ** 2))),
        "r2": float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "corr": float(np.corrcoef(sim, obs)[0, 1]),
    }


def fit_lotka_volterra(df: pd.DataFrame, n_train: Optional[int] = None, n_starts: int = 60, seed: int = 0,
                       weighted: bool = True) -> FitResult:
    """
    Multi-start least-squares fit on the first n_train points (all points by default).

    weighted=True divides each species by its standard deviation (equal weight for both);
    weighted=False minimises the plain sum of squared errors in data units.
    """
    n_train = len(df) if n_train is None else n_train
    t = (df["year"].to_numpy(dtype=float) - df["year"].iloc[0])[:n_train]
    hare = df["hare"].to_numpy(dtype=float)[:n_train]
    lynx = df["lynx"].to_numpy(dtype=float)[:n_train]
    s_hare, s_lynx = (hare.std(), lynx.std()) if weighted else (1.0, 1.0)

    def residuals(theta):
        sol = _solve(theta, t)
        if not sol.success or sol.y.shape[1] != len(t):
            return np.full(2 * len(t), 1e3)
        return np.concatenate([(sol.y[0] - hare) / s_hare, (sol.y[1] - lynx) / s_lynx])

    rng = np.random.default_rng(seed)
    runs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_starts):
            theta0 = np.log([
                rng.uniform(0.2, 1.5),        # alpha
                10 ** rng.uniform(-3, -1),    # beta
                10 ** rng.uniform(-3, -1),    # delta
                rng.uniform(0.2, 1.5),        # gamma
                hare[0], lynx[0],             # x0, y0 start at the first observation
            ])
            try:
                res = least_squares(residuals, np.clip(theta0, _LOWER, _UPPER), bounds=(_LOWER, _UPPER), max_nfev=300)
            except Exception:  # a start that makes the solver fail is simply discarded
                continue
            runs.append((2 * res.cost / len(res.fun), res.x))

    if not runs:
        raise RuntimeError("No optimisation run succeeded.")
    runs.sort(key=lambda r: r[0])
    cost, theta = runs[0]
    params = dict(zip(PARAM_NAMES, np.exp(theta).tolist()))
    sim = simulate_fit(theta, t)
    return FitResult(
        params=params, theta=theta, cost=float(cost),
        sse=float(np.sum((sim[0] - hare) ** 2) + np.sum((sim[1] - lynx) ** 2)), n_starts=len(runs),
        n_near_best=int(sum(r[0] <= 1.05 * cost for r in runs)),
        period=estimate_period(theta),
        equilibrium=(params["gamma"] / params["delta"], params["alpha"] / params["beta"]),
        metrics={"hare": _metrics(sim[0], hare), "lynx": _metrics(sim[1], lynx)},
    )


def holdout_check(df: pd.DataFrame, n_train: int = 13, n_starts: int = 40, seed: int = 0) -> Dict:
    """Fits on the first n_train years and evaluates the forecast on the remaining years."""
    fit = fit_lotka_volterra(df, n_train=n_train, n_starts=n_starts, seed=seed)
    t = df["year"].to_numpy(dtype=float) - df["year"].iloc[0]
    sim = simulate_fit(fit.theta, t)
    test = slice(n_train, len(df))
    out = {"fit": fit, "sim": sim, "n_train": n_train}
    for i, name in enumerate(["hare", "lynx"]):
        obs = df[name].to_numpy(dtype=float)
        out[name] = {
            "rmse": float(np.sqrt(np.mean((sim[i][test] - obs[test]) ** 2))),
            "corr": float(np.corrcoef(sim[i][test], obs[test])[0, 1]),
            # reference: always forecasting the training-period mean
            "rmse_mean_baseline": float(np.sqrt(np.mean((obs[:n_train].mean() - obs[test]) ** 2))),
        }
    return out


def plot_fit(df: pd.DataFrame, fit: FitResult, holdout: Dict, save_path: Path):
    import matplotlib.pyplot as plt

    years = df["year"].to_numpy()
    t = years - years[0]
    t_fine = np.linspace(t[0], t[-1], 400)
    fine = simulate_fit(fit.theta, t_fine)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(years, df["hare"], "o", color="#1f77b4", label="Hare (observed)")
    ax.plot(years, df["lynx"], "s", color="#d62728", label="Lynx (observed)")
    ax.plot(years[0] + t_fine, fine[0], "-", color="#1f77b4", label="Hare (fitted)")
    ax.plot(years[0] + t_fine, fine[1], "-", color="#d62728", label="Lynx (fitted)")
    ax.set_ylabel("Pelts traded (thousands)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Fit on all 21 years - period {fit.period:.1f} years, "
                 f"R2 hare {fit.metrics['hare']['r2']:.2f} / lynx {fit.metrics['lynx']['r2']:.2f}")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    n = holdout["n_train"]
    sim_h = simulate_fit(holdout["fit"].theta, t_fine)
    ax.plot(years, df["hare"], "o", color="#1f77b4", label="Hare (observed)")
    ax.plot(years, df["lynx"], "s", color="#d62728", label="Lynx (observed)")
    ax.plot(years[0] + t_fine, sim_h[0], "--", color="#1f77b4", label="Hare (fit on 1900-%d)" % years[n - 1])
    ax.plot(years[0] + t_fine, sim_h[1], "--", color="#d62728", label="Lynx (fit on 1900-%d)" % years[n - 1])
    ax.axvspan(years[n - 1] + 0.5, years[-1] + 0.5, color="grey", alpha=0.15, label="Forecast period")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pelts traded (thousands)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Hold-out check - forecast RMSE {holdout['hare']['rmse']:.1f} (hare), {holdout['lynx']['rmse']:.1f} (lynx)")
    ax.legend(ncol=3, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Fit the Lotka-Volterra model to the Hudson Bay lynx-hare data.")
    parser.add_argument("--starts", type=int, default=60, help="number of random restarts (default: 60)")
    parser.add_argument("--show", action="store_true", help="display the figure window")
    parser.add_argument("--unweighted", action="store_true",
                        help="plain sum of squared errors in data units (no hold-out, no figure)")
    args = parser.parse_args()

    df = load_real_data()
    print(f"Fitting with {args.starts} random starts...")
    fit = fit_lotka_volterra(df, n_starts=args.starts, weighted=not args.unweighted)
    p = fit.params
    print(f"alpha={p['alpha']:.4f}/yr  gamma={p['gamma']:.4f}/yr  beta={p['beta']:.5f}  delta={p['delta']:.5f}  "
          f"x0={p['x0']:.2f}  y0={p['y0']:.2f}")
    print(f"Mean squared {'residual' if args.unweighted else 'normalised residual'}: {fit.cost:.4f} "
          f"({fit.n_near_best}/{fit.n_starts} starts reached the best solution)")
    print(f"Sum of squared errors: {fit.sse:.2f} (thousand pelts^2)")
    print(f"Fitted cycle period: {fit.period:.2f} years")
    print(f"Equilibrium (hare, lynx): ({fit.equilibrium[0]:.1f}, {fit.equilibrium[1]:.1f}) thousand pelts "
          f"| observed means: ({df['hare'].mean():.1f}, {df['lynx'].mean():.1f})")
    for sp, m in fit.metrics.items():
        print(f"  {sp}: RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}  corr={m['corr']:.3f}")

    if args.unweighted:
        return

    ho = holdout_check(df)
    print(f"\nHold-out: fit on {df['year'].iloc[0]}-{df['year'].iloc[ho['n_train'] - 1]}, forecast the following years")
    for sp in ("hare", "lynx"):
        h = ho[sp]
        print(f"  {sp}: forecast RMSE={h['rmse']:.2f} (training-mean baseline {h['rmse_mean_baseline']:.2f}), corr={h['corr']:.3f}")

    out = FIGURES_DIR / "real_vs_fitted.png"
    plot_fit(df, fit, ho, out)
    print(f"\nFigure saved to {out}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
