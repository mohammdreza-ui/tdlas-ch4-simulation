"""Allan-Werle variance and noise budget analysis for laser absorption spectroscopy.

Formula:
    sigma_A^2(tau) = 1 / (2 * (M - 1)) * sum_{k=1}^{M-1} (bar_x_{k+1} - bar_x_k)^2
References:
    Werle et al. (1993), Near and mid-infrared laser-optical sensors for gas analysis.
"""
from __future__ import annotations
import numpy as np

def compute_allan_deviation(
    y: np.ndarray, rate_hz: float, taus: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute overlapping Allan deviation sigma_A(tau)."""
    N = len(y)
    dt = 1.0 / rate_hz
    
    if taus is None:
        max_m = N // 10
        m_factors = np.unique(np.logspace(0, np.log10(max_m), 30).astype(int))
    else:
        m_factors = np.unique((taus / dt).astype(int))
        m_factors = m_factors[(m_factors >= 1) & (m_factors <= N // 4)]
        
    actual_taus = []
    allan_devs = []
    
    for m in m_factors:
        tau = m * dt
        # Bin and average into non-overlapping blocks of size m
        n_blocks = N // m
        if n_blocks < 3:
            continue
            
        reshaped = y[: n_blocks * m].reshape((n_blocks, m))
        block_means = np.mean(reshaped, axis=1)
        
        diff = np.diff(block_means)
        if len(diff) < 2:
            continue
            
        allan_var = 0.5 * np.mean(diff ** 2)
        actual_taus.append(tau)
        allan_devs.append(float(np.sqrt(allan_var)))
        
    return np.array(actual_taus), np.array(allan_devs)


def build_synthetic_series(
    n_pts: int,
    fs_hz: float,
    *,
    seed: int = 42,
    white_std: float = 0.002,
    pink_rms: float = 0.0008,
    drift_per_s: float = 0.0003,
) -> np.ndarray:
    """Hand-built test series used for Figure 3: white noise + pink (1/f power) noise + linear drift.

    This is NOT a series of fitted concentrations; the 1/f and drift terms are added by hand.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, white_std, n_pts)
    freqs = np.fft.rfftfreq(n_pts, d=1.0 / fs_hz)
    freqs[0] = 1e-6
    pink_spec = (1.0 / np.sqrt(freqs)) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(freqs)))
    pink = np.fft.irfft(pink_spec, n=n_pts)
    pink = pink * (pink_rms / np.std(pink))
    drift = drift_per_s * (np.arange(n_pts) / fs_hz)
    return white + pink + drift
