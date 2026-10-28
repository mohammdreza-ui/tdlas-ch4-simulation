"""From-scratch Levenberg-Marquardt non-linear least squares solver with analytical Jacobians.

Model:
    V_model(t, nu; x, c_0, ..., c_K) = (sum_{k=0}^K c_k * t^k) * exp(-A(nu; x))
Residual:
    r_i = V_model(t_i) - V_meas(t_i)
"""
from __future__ import annotations
import numpy as np
from tdlas.absorption import compute_absorption_coefficient

def compute_residuals_and_jacobian(
    theta: np.ndarray,
    nu_axis_cm1: np.ndarray,
    t_norm: np.ndarray,
    V_measured: np.ndarray,
    T_K: float,
    P_atm: float,
    L_cm: float,
    lines_data: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute residual vector r and analytical Jacobian matrix J = dr / dtheta.
    
    theta = [x_ppm, c_0, c_1, ..., c_K]
    """
    x_ppm = float(theta[0])
    poly_coeffs = theta[1:]  # c_0 + c_1*t + c_2*t^2 + ...
    
    # Powers of t: [N, K+1]
    K = len(poly_coeffs)
    T_powers = np.column_stack([t_norm ** k for k in range(K)])
    
    # Baseline polynomial P_L(t)
    baseline = T_powers @ poly_coeffs
    
    # Absorption: A(x) = alpha(x) * L
    # alpha is strictly linear in x: alpha(x) = x_ppm * alpha_unit
    alpha_1ppm = compute_absorption_coefficient(nu_axis_cm1, 1.0, T_K, P_atm, lines_data)
    absorbance = x_ppm * alpha_1ppm * L_cm
    transmission = np.exp(-absorbance)
    
    # Model output
    V_model = baseline * transmission
    residuals = V_model - V_measured
    
    # Jacobian dr / dtheta
    N = len(residuals)
    J = np.zeros((N, len(theta)), dtype=np.float64)
    
    # Partial derivative w.r.t x_ppm:
    # dV / dx = baseline * transmission * (-alpha_1ppm * L_cm)
    J[:, 0] = baseline * transmission * (-alpha_1ppm * L_cm)
    
    # Partial derivatives w.r.t baseline coefficients c_k:
    # dV / dc_k = t^k * transmission
    for k in range(K):
        J[:, 1 + k] = T_powers[:, k] * transmission
        
    return residuals, J

def levenberg_marquardt_fit(
    V_measured: np.ndarray,
    nu_axis_cm1: np.ndarray,
    t_norm: np.ndarray,
    T_K: float,
    P_atm: float,
    L_cm: float,
    lines_data: list[dict],
    poly_order: int = 3,
    initial_x_ppm: float = 10.0,
    max_iter: int = 50,
    tol: float = 1e-7
) -> dict:
    """Execute Levenberg-Marquardt non-linear least squares fit."""
    n = len(V_measured)
    mask_wings = np.zeros(n, dtype=bool)
    mask_wings[: int(0.15 * n)] = True
    mask_wings[int(0.85 * n) :] = True
    
    T_powers = np.column_stack([t_norm ** k for k in range(poly_order + 1)])
    c_init, _, _, _ = np.linalg.lstsq(T_powers[mask_wings], V_measured[mask_wings], rcond=None)
    
    theta = np.zeros(1 + len(c_init), dtype=np.float64)
    theta[0] = initial_x_ppm
    theta[1:] = c_init
    initial_theta = theta.copy()
    
    lam = 1e-2
    r, J = compute_residuals_and_jacobian(
        theta, nu_axis_cm1, t_norm, V_measured, T_K, P_atm, L_cm, lines_data
    )
    current_cost = 0.5 * np.sum(r ** 2)
    converged = False
    
    for it in range(max_iter):
        JtJ = J.T @ J
        Jtr = J.T @ r
        
        diag_JtJ = np.diag(np.diag(JtJ))
        H_damped = JtJ + lam * (diag_JtJ + 1e-6 * np.eye(len(theta)))
        
        try:
            delta = np.linalg.solve(H_damped, -Jtr)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
            
        theta_candidate = theta + delta
        theta_candidate[0] = max(0.0, float(theta_candidate[0]))
        
        r_cand, J_cand = compute_residuals_and_jacobian(
            theta_candidate, nu_axis_cm1, t_norm, V_measured, T_K, P_atm, L_cm, lines_data
        )
        new_cost = 0.5 * np.sum(r_cand ** 2)
        
        if new_cost < current_cost:
            lam = max(lam / 10.0, 1e-7)
            cost_drop = current_cost - new_cost
            theta = theta_candidate
            r = r_cand
            J = J_cand
            current_cost = new_cost
            if cost_drop < tol or np.linalg.norm(delta) < tol:
                converged = True
                break
        else:
            lam = min(lam * 10.0, 1e7)
            
    return {
        "x_ppm": float(theta[0]),
        "baseline_coeffs": theta[1:],
        "converged": converged,
        "iterations": it + 1,
        "final_cost": float(current_cost),
        "residuals": r,
        "jacobian": J,
        "initial_theta": initial_theta
    }
