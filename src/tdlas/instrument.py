"""Simulated TDLAS instrument model: laser ramp, intensity profile, optical etalon, noise, and ADC.

Formula:
    V(t) = G * P_L(i(t)) * [1 + epsilon_et(t)] * exp(-A(nu(t))) + n(t)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tdlas.absorption import compute_transmission

@dataclass
class DeviceParams:
    """Hardware parameters characterizing a simulated physical TDLAS sensor."""
    device_id: str
    n_points: int = 1000
    scan_time_s: float = 0.01  # 100 Hz scan repetition
    # Wavelength tuning: nu(t) = nu_center + tuning_linear * norm_t + tuning_quad * norm_t^2
    nu_center_cm1: float = 3018.15
    tuning_linear_cm1: float = 1.2
    tuning_quad_cm1: float = 0.05
    # Optical path length in integrating sphere / multipass cell [cm]
    path_length_cm: float = 70.0
    # Laser power polynomial coefficients: P(norm_t) = p0 + p1*norm_t + p2*norm_t^2
    p0: float = 2.2
    p1: float = 1.2
    p2: float = -0.2
    # Optical etalon fringe parameters: m * cos(2*pi * nu / FSR + phi)
    etalon_amplitude: float = 0.008   # 0.8% fringe
    etalon_fsr_cm1: float = 0.12     # Free Spectral Range
    etalon_phase_rad: float = 0.5
    # Noise components
    white_noise_std_v: float = 1.5e-3
    flicker_noise_std_v: float = 8.0e-4
    drift_rate_v_s: float = 0.005
    # ADC characteristics
    adc_bits: int = 16
    voltage_range_v: float = 5.0

    @classmethod
    def sample_random(cls, device_id: str, seed: int | None = None) -> DeviceParams:
        """Draw a randomized hardware parameter set representing an individual sensor instance."""
        rng = np.random.default_rng(seed)
        return cls(
            device_id=device_id,
            nu_center_cm1=float(rng.uniform(3018.10, 3018.20)),
            tuning_linear_cm1=float(rng.uniform(1.10, 1.30)),
            tuning_quad_cm1=float(rng.uniform(0.02, 0.08)),
            path_length_cm=float(rng.uniform(68.0, 72.0)),
            p0=float(rng.uniform(2.0, 2.5)),
            p1=float(rng.uniform(0.9, 1.5)),
            p2=float(rng.uniform(-0.3, -0.1)),
            etalon_amplitude=float(rng.uniform(0.003, 0.015)),
            etalon_fsr_cm1=float(rng.uniform(0.09, 0.15)),
            etalon_phase_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
            white_noise_std_v=float(rng.uniform(1.0e-3, 2.5e-3)),
            flicker_noise_std_v=float(rng.uniform(4.0e-4, 1.2e-3)),
            adc_bits=16,
            voltage_range_v=5.0
        )

def generate_1f_noise(
    n_samples: int, sample_rate_hz: float, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate colored 1/f^alpha noise via frequency-domain filtering."""
    white = rng.normal(0.0, 1.0, n_samples)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    freqs[0] = freqs[1]  # avoid division by zero at DC
    scale = 1.0 / (freqs ** (alpha / 2.0))
    colored_fft = fft_white * scale
    noise = np.fft.irfft(colored_fft, n=n_samples)
    std = np.std(noise)
    return (noise / (std if std > 1e-12 else 1.0))

def simulate_scan(
    x_ppm: float,
    T_K: float,
    P_atm: float,
    device: DeviceParams,
    rng: np.random.Generator,
    lines_data: list[dict],
    include_noise: bool = True,
    extra_lines: list[dict] | None = None,
    extra_x_ppm: float = 0.0
) -> dict:
    """Generate a synthetic TDLAS detector voltage scan.

    extra_lines / extra_x_ppm optionally add an interfering absorber (computed with the same
    line-shape machinery as CH4) that the CH4-only forward model of the fit does not know about.
    """
    n = device.n_points
    t_axis_s = np.linspace(0.0, device.scan_time_s, n)
    norm_t = np.linspace(-0.5, 0.5, n)
    
    # 1. Optical wavelength tuning
    nu_axis_cm1 = (
        device.nu_center_cm1
        + device.tuning_linear_cm1 * norm_t
        + device.tuning_quad_cm1 * (norm_t ** 2)
    )
    
    # 2. Raw Laser Output Power (baseline)
    laser_power_v = device.p0 + device.p1 * norm_t + device.p2 * (norm_t ** 2)
    
    # 3. Optical etalon fringe
    etalon_fringe = 1.0 + device.etalon_amplitude * np.cos(
        2.0 * np.pi * nu_axis_cm1 / device.etalon_fsr_cm1 + device.etalon_phase_rad
    )
    
    # Dedicated low-noise etalon calibration channel (reference Fabry-Perot interferometer)
    etalon_ref_channel = 2.0 * (1.0 + 0.5 * np.cos(
        2.0 * np.pi * nu_axis_cm1 / device.etalon_fsr_cm1
    ))
    
    # 4. Physical gas transmission
    absorbance, transmission = compute_transmission(
        nu_axis_cm1, x_ppm, T_K, P_atm, device.path_length_cm, lines_data
    )
    
    if extra_lines and extra_x_ppm > 0.0:
        _, extra_transmission = compute_transmission(
            nu_axis_cm1, extra_x_ppm, T_K, P_atm, device.path_length_cm, extra_lines
        )
        transmission = transmission * extra_transmission

    signal_v = laser_power_v * etalon_fringe * transmission

    # 5. Noise addition
    if include_noise:
        sample_rate = n / device.scan_time_s
        white_noise = rng.normal(0.0, device.white_noise_std_v, n)
        flicker_noise = device.flicker_noise_std_v * generate_1f_noise(n, sample_rate, 1.0, rng)
        # Slow baseline random-walk drift
        drift = np.cumsum(rng.normal(0.0, device.drift_rate_v_s / np.sqrt(n), n))
        total_voltage = signal_v + white_noise + flicker_noise + drift
        etalon_ref_channel += rng.normal(0.0, 1.0e-3, n)
    else:
        total_voltage = signal_v
        
    # 6. ADC quantization
    lsb = device.voltage_range_v / (2 ** device.adc_bits)
    clipped_v = np.clip(total_voltage, 0.0, device.voltage_range_v)
    quantized_v = np.round(clipped_v / lsb) * lsb
    
    return {
        "time_s": t_axis_s,
        "nu_axis_cm1": nu_axis_cm1,
        "voltage_v": quantized_v,
        "etalon_channel_v": etalon_ref_channel,
        "ground_truth": {
            "x_ppm": x_ppm,
            "T_K": T_K,
            "P_atm": P_atm,
            "absorbance": absorbance,
            "laser_baseline_v": laser_power_v,
            "etalon_fringe": etalon_fringe,
            "device_id": device.device_id
        }
    }
