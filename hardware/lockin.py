"""Software dual-phase lock-in amplifier for modulated Beer-Lambert optical testing.

Multiplies input with in-phase (cos) and quadrature (sin) reference signals,
then averages over an integer number of modulation cycles.
"""
from __future__ import annotations
import numpy as np

def software_lockin_detector(
    signal_v: np.ndarray, ref_freq_hz: float, sample_rate_hz: float
) -> tuple[float, float]:
    """Demodulate input signal to extract amplitude and phase."""
    N = len(signal_v)
    t = np.arange(N) / sample_rate_hz
    
    # Reference waves
    ref_cos = np.cos(2.0 * np.pi * ref_freq_hz * t)
    ref_sin = np.sin(2.0 * np.pi * ref_freq_hz * t)
    
    # Truncate to exact integer number of cycles
    samples_per_cycle = sample_rate_hz / ref_freq_hz
    n_cycles = int(N / samples_per_cycle)
    if n_cycles < 1:
        raise ValueError("Signal length too short for one modulation cycle")
    valid_len = int(n_cycles * samples_per_cycle)
    
    I = 2.0 * np.mean(signal_v[:valid_len] * ref_cos[:valid_len])
    Q = -2.0 * np.mean(signal_v[:valid_len] * ref_sin[:valid_len])
    
    amplitude = float(np.sqrt(I ** 2 + Q ** 2))
    phase_rad = float(np.arctan2(Q, I))
    return amplitude, phase_rad
