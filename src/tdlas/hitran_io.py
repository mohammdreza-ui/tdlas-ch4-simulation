"""Offline HITRAN line parameter loading and Total Internal Partition Sums (TIPS) for CH4.

References:
- Gamache et al. (2017), Total internal partition sums for 166 isotopologues of 51 molecules.
- Rothman et al., The HITRAN molecular spectroscopic database.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# Physical constants
C2_CM_K = 1.4387769  # Second radiation constant h*c/k_B [cm * K]
T0_REF_K = 296.0     # HITRAN standard reference temperature [K]

DATA_DIR = Path(__file__).resolve().parent / "data"

# Fallback in-memory catalog
CH4_3018_LINES = [
    {
        "nu_cm1": 3017.414,
        "S_T0": 1.15e-19,      # cm^-1 / (molecule * cm^-2)
        "gamma_air": 0.0620,   # cm^-1 / atm
        "gamma_self": 0.0780,  # cm^-1 / atm
        "n_air": 0.72,
        "delta_air": -0.0085,  # cm^-1 / atm
        "elower": 62.87        # cm^-1
    },
    {
        "nu_cm1": 3017.982,
        "S_T0": 8.92e-20,
        "gamma_air": 0.0590,
        "gamma_self": 0.0740,
        "n_air": 0.70,
        "delta_air": -0.0090,
        "elower": 105.12
    },
    {
        "nu_cm1": 3018.423,
        "S_T0": 1.84e-19,
        "gamma_air": 0.0645,
        "gamma_self": 0.0810,
        "n_air": 0.75,
        "delta_air": -0.0080,
        "elower": 31.44
    },
    {
        "nu_cm1": 3018.895,
        "S_T0": 4.50e-20,
        "gamma_air": 0.0560,
        "gamma_self": 0.0710,
        "n_air": 0.68,
        "delta_air": -0.0095,
        "elower": 150.25
    }
]

def partition_function_ch4(T_K: float) -> float:
    """Compute Total Internal Partition Sum Q(T) for 12CH4.
    
    Uses standard polynomial interpolation fitted to HITRAN TIPS data.
    Q(T) = a + b*T + c*T^2 + d*T^3 (valid for 150 K <= T <= 500 K).
    """
    if T_K <= 0:
        raise ValueError(f"Temperature T_K must be positive, got {T_K}")
    t = float(T_K)
    # TIPS polynomial coefficients fitted to HITRAN TIPS table for 12CH4
    # At T=296 K, Q(296) = 590.41
    a = -2.9874e+02
    b = 4.8918e+00
    c = -1.2444e-02
    d = 2.0494e-05
    return float(a + b * t + c * (t ** 2) + d * (t ** 3))

def temperature_adjust_intensity(
    S_T0: float, nu0_cm1: float, elower_cm1: float, T_K: float
) -> float:
    """Adjust line intensity S(T0) to temperature T_K using the HITRAN formulation.
    
    Formula:
    S(T) = S(T0) * (Q(T0) / Q(T)) * exp(-c2 * E'' / T) / exp(-c2 * E'' / T0)
           * [1 - exp(-c2 * nu0 / T)] / [1 - exp(-c2 * nu0 / T0)]
    """
    q_ratio = partition_function_ch4(T0_REF_K) / partition_function_ch4(T_K)
    boltzmann_factor = np.exp(-C2_CM_K * elower_cm1 * (1.0 / T_K - 1.0 / T0_REF_K))
    stim_emission_T = 1.0 - np.exp(-C2_CM_K * nu0_cm1 / T_K)
    stim_emission_T0 = 1.0 - np.exp(-C2_CM_K * nu0_cm1 / T0_REF_K)
    return float(S_T0 * q_ratio * boltzmann_factor * (stim_emission_T / stim_emission_T0))

def load_hitran_ch4_window(nu_min_cm1: float, nu_max_cm1: float) -> list[dict]:
    """Retrieve CH4 absorption lines falling within [nu_min_cm1, nu_max_cm1] completely offline."""
    json_path = DATA_DIR / "hitran_ch4_3018.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            all_lines = json.load(f)
    else:
        all_lines = CH4_3018_LINES
        
    return [
        line for line in all_lines
        if nu_min_cm1 <= line["nu_cm1"] <= nu_max_cm1
    ]
