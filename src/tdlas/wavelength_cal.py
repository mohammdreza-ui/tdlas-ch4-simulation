"""Wavelength axis calibration using Fabry-Perot etalon interference fringes."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks

def calibrate_wavelength_axis(
    t_s: np.ndarray,
    etalon_v: np.ndarray,
    fsr_cm1: float,
    nu_anchor_cm1: float,
    poly_order: int = 2
) -> np.ndarray:
    """Reconstruct absolute optical frequency axis nu(t) from etalon fringes.
    
    Finds fringe peaks and troughs, maps each consecutive fringe to an FSR step,
    and fits a polynomial nu(t) anchored at the known reference nu_anchor_cm1.
    """
    detrended = etalon_v - np.mean(etalon_v)
    
    # Distance of at least 15 samples between extrema
    peaks, _ = find_peaks(detrended, distance=15)
    troughs, _ = find_peaks(-detrended, distance=15)
    
    all_extrema = np.sort(np.concatenate([peaks, troughs]))
    if len(all_extrema) < 4:
        # Fallback if fringe count is too low
        return np.linspace(nu_anchor_cm1 - 0.6, nu_anchor_cm1 + 0.6, len(t_s))
        
    t_extrema = t_s[all_extrema]
    # Each transition peak <-> trough is a half FSR = 0.5 * fsr_cm1
    relative_nu = np.arange(len(all_extrema), dtype=np.float64) * (0.5 * fsr_cm1)
    
    # Anchor at the midpoint
    mid_idx = len(all_extrema) // 2
    t_mid = t_extrema[mid_idx]
    relative_nu -= relative_nu[mid_idx]
    absolute_nu = nu_anchor_cm1 + relative_nu
    
    # Fit polynomial nu(t)
    poly_coeffs = np.polyfit(t_extrema, absolute_nu, deg=poly_order)
    nu_calibrated = np.polyval(poly_coeffs, t_s)
    
    return nu_calibrated
