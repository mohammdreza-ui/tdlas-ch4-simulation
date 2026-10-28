"""Forward model for methane (CH4) optical absorption under Beer-Lambert law.

References:
- Bernath, Spectra of Atoms and Molecules, Oxford University Press.
- Hodgkinson et al. (2023), Cranfield University methane sensor group.
"""
from __future__ import annotations
import numpy as np
from tdlas.hitran_io import temperature_adjust_intensity, T0_REF_K
from tdlas.lineshape import voigt_profile

# Physical constants
KB_J_K = 1.380649e-23     # Boltzmann constant [J / K]
ATM_TO_PA = 101325.0       # 1 atm in Pascals [N / m^2]
SPEED_OF_LIGHT_CM_S = 2.99792458e10  # Speed of light [cm / s]
MOLAR_MASS_CH4_KG = 16.0425e-3 / 6.02214076e23  # Mass of single CH4 molecule [kg]

def compute_number_density(x_ppm: float, P_atm: float, T_K: float) -> float:
    """Compute methane molecular number density N [molecules / cm^3].
    
    Formula:
        N = (x * P) / (k_B * T)
        where x = x_ppm * 1e-6, P in Pa, T in K. Convert m^-3 to cm^-3 (* 1e-6).
    """
    x_mole_fraction = x_ppm * 1e-6
    P_pa = P_atm * ATM_TO_PA
    N_m3 = (x_mole_fraction * P_pa) / (KB_J_K * T_K)
    return float(N_m3 * 1e-6)

def compute_doppler_hwhm(nu0_cm1: float, T_K: float) -> float:
    """Compute Doppler half-width at half-maximum gamma_D [cm^-1].
    
    Formula:
        gamma_D = (nu0 / c) * sqrt(2 * k_B * T * ln(2) / m)
    """
    v_th = np.sqrt(2.0 * KB_J_K * T_K * np.log(2.0) / MOLAR_MASS_CH4_KG)
    v_th_cm_s = v_th * 100.0
    return float((nu0_cm1 / SPEED_OF_LIGHT_CM_S) * v_th_cm_s)

def compute_lorentz_hwhm(
    P_atm: float, T_K: float, x_ppm: float, gamma_air: float, gamma_self: float, n_air: float
) -> float:
    """Compute Lorentz (pressure-broadened) half-width gamma_L [cm^-1].
    
    Formula:
        gamma_L = (T0 / T)^n * [gamma_air * (P - p_s) + gamma_self * p_s]
        where p_s = x * P is methane partial pressure.
    """
    x_mole_fraction = x_ppm * 1e-6
    p_s = x_mole_fraction * P_atm
    t_factor = (T0_REF_K / T_K) ** n_air
    return float(t_factor * (gamma_air * (P_atm - p_s) + gamma_self * p_s))

def compute_absorption_coefficient(
    nu_axis_cm1: np.ndarray,
    x_ppm: float,
    T_K: float,
    P_atm: float,
    lines_data: list[dict]
) -> np.ndarray:
    """Compute total linear absorption coefficient alpha(nu) [cm^-1].
    
    Formula:
        alpha(nu) = sum_i S_i(T) * N * g_i(nu - nu_i)
    """
    N_ch4 = compute_number_density(x_ppm, P_atm, T_K)
    alpha = np.zeros_like(nu_axis_cm1, dtype=np.float64)
    
    for line in lines_data:
        # Pressure-shifted line center: nu0 + delta_air * P
        nu0_shifted = line["nu_cm1"] + line["delta_air"] * P_atm
        # Temperature-adjusted line intensity S(T)
        S_T = temperature_adjust_intensity(line["S_T0"], line["nu_cm1"], line["elower"], T_K)
        # Doppler and Lorentz widths
        gamma_D = compute_doppler_hwhm(nu0_shifted, T_K)
        gamma_L = compute_lorentz_hwhm(
            P_atm, T_K, x_ppm, line["gamma_air"], line["gamma_self"], line["n_air"]
        )
        # Normalized Voigt profile g(nu - nu0)
        g = voigt_profile(nu_axis_cm1, nu0_shifted, gamma_D, gamma_L)
        alpha += S_T * N_ch4 * g
        
    return alpha

def compute_transmission(
    nu_axis_cm1: np.ndarray,
    x_ppm: float,
    T_K: float,
    P_atm: float,
    L_cm: float,
    lines_data: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute absorbance A(nu) and optical transmission T(nu) = exp(-A(nu)).
    
    Beer-Lambert Law:
        A(nu) = alpha(nu) * L_cm
        I(nu) / I_0(nu) = exp(-A(nu))
    """
    alpha = compute_absorption_coefficient(nu_axis_cm1, x_ppm, T_K, P_atm, lines_data)
    absorbance = alpha * L_cm
    transmission = np.exp(-absorbance)
    return absorbance, transmission
