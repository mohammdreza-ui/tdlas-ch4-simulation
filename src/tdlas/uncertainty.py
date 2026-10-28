"""Statistical uncertainty quantification, Fisher Information, and Cramér-Rao Lower Bound (CRB).

Formulas:
    Cov(theta) ~= sigma^2 * (J^T * J)^(-1)
    Var(x) >= [I(theta)^(-1)]_xx, where I(theta) = (J^T * J) / sigma^2
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from tdlas.fit import levenberg_marquardt_fit
from tdlas.instrument import DeviceParams, simulate_scan

def compute_covariance_and_crb(
    J: np.ndarray, sigma_noise: float
) -> tuple[np.ndarray, float]:
    """Compute asymptotic parameter covariance matrix and CRB standard deviation for concentration x."""
    JtJ = J.T @ J
    # Invert full Fisher / normal matrix using pseudo-inverse for numerical robustness
    cov_matrix = (sigma_noise ** 2) * np.linalg.pinv(JtJ)
    var_x = float(cov_matrix[0, 0])
    crb_std_x = float(np.sqrt(max(var_x, 1e-15)))
    return cov_matrix, crb_std_x

def verify_confidence_coverage(
    fitted_x: np.ndarray,
    true_x: float,
    sigmas_x: np.ndarray,
    confidence: float = 0.95
) -> float:
    """Compute empirical coverage proportion for nominal (1 - alpha) confidence interval."""
    alpha = 1.0 - confidence
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    
    lower_bound = fitted_x - z_crit * sigmas_x
    upper_bound = fitted_x + z_crit * sigmas_x
    
    covered = (true_x >= lower_bound) & (true_x <= upper_bound)
    return float(np.mean(covered))


def run_monte_carlo(
    x_ppm: float,
    n_scans: int,
    lines_data: list[dict],
    device: DeviceParams,
    *,
    seed: int = 11,
    T_K: float = 296.0,
    P_atm: float = 1.0,
    initial_x_ppm: float = 5.0,
) -> dict:
    """Simulate and fit n_scans scans of one device and compare the scatter with the CRB.

    sigma for the CRB is the standard deviation of each fit's residuals (as in the rest of the repo).
    Returns the mean estimate, bias, empirical std, mean CRB std, their ratio, 95% CI coverage,
    the non-convergence rate and the min/max iteration count.
    """
    rng = np.random.default_rng(seed)
    t_norm = np.linspace(-0.5, 0.5, device.n_points)
    estimates, crb_stds, iterations, failures = [], [], [], 0
    for _ in range(n_scans):
        scan = simulate_scan(x_ppm, T_K, P_atm, device, rng, lines_data)
        fit = levenberg_marquardt_fit(
            scan["voltage_v"], scan["nu_axis_cm1"], t_norm, T_K, P_atm, device.path_length_cm,
            lines_data, initial_x_ppm=initial_x_ppm,
        )
        failures += 0 if fit["converged"] else 1
        iterations.append(fit["iterations"])
        _, crb = compute_covariance_and_crb(fit["jacobian"], float(np.std(fit["residuals"])))
        estimates.append(fit["x_ppm"])
        crb_stds.append(crb)
    est, crb = np.array(estimates), np.array(crb_stds)
    std = float(np.std(est, ddof=1))
    return {
        "x_ppm": x_ppm, "n_scans": n_scans,
        "mean_estimate": float(np.mean(est)), "bias": float(np.mean(est) - x_ppm),
        "std_empirical": std, "mean_crb_std": float(np.mean(crb)),
        "crb_over_empirical": float(np.mean(crb) / std),
        "coverage_95": verify_confidence_coverage(est, x_ppm, crb),
        "nonconverged_rate": failures / n_scans,
        "iterations_min": int(min(iterations)), "iterations_max": int(max(iterations)),
    }
