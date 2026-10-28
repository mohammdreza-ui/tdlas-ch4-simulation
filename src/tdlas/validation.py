"""Numerical checks whose results are quoted in the README and the technical report.

Each value comes from a seeded run of the code in this repository. The checks are:
    - Faddeeva / Voigt accuracy against scipy.special.wofz and the area normalisation
    - least-squares fit recovery on noiseless data, and agreement with scipy.optimize.least_squares
    - Monte Carlo of the fit (bias, scatter, CRB, 95% coverage) with white noise only and with the
      default instrument noise (white + 1/f + drift + etalon)
    - Allan deviation slope for white noise and the statistics of the hand-built Figure 3 series
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.special import wofz
from tdlas.absorption import compute_absorption_coefficient
from tdlas.allan import compute_allan_deviation, build_synthetic_series
from tdlas.fit import levenberg_marquardt_fit
from tdlas.hitran_io import partition_function_ch4
from tdlas.instrument import DeviceParams, simulate_scan
from tdlas.lineshape import faddeeva_rational, voigt_profile
from tdlas.uncertainty import run_monte_carlo
from tdlas.wavelength_cal import calibrate_wavelength_axis

MC_DEVICE_SEED = 5
IDEAL_NOISE = dict(etalon_amplitude=0.0, flicker_noise_std_v=0.0, drift_rate_v_s=0.0)


def lineshape_checks() -> dict:
    x = np.linspace(-50.0, 50.0, 801)
    y = np.logspace(-4, 2, 60)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    wide = np.max(np.abs(faddeeva_rational(Z) - wofz(Z)) / np.abs(wofz(Z)))
    # z = sqrt(ln2) * (dnu + i*gamma_L) / gamma_D for |dnu| <= 2 cm^-1, gamma_L = 0.062, gamma_D = 0.0046
    z_ch4 = np.sqrt(np.log(2.0)) * (np.linspace(-2.0, 2.0, 2001) + 1j * 0.062) / 0.0046
    ch4 = np.max(np.abs(faddeeva_rational(z_ch4) - wofz(z_ch4)) / np.abs(wofz(z_ch4)))
    nu = np.linspace(2900.0, 3136.0, 100000)
    area = float(np.trapezoid(voigt_profile(nu, 3018.0, 0.015, 0.035), nu))
    return {
        "faddeeva_max_rel_err_wide_grid": float(wide),
        "faddeeva_max_rel_err_ch4_grid": float(ch4),
        "voigt_area_236cm1_window": area,
        "partition_function_296K": partition_function_ch4(296.0),
    }


def fit_checks(lines: list[dict], n_scipy_scans: int) -> dict:
    t_norm = np.linspace(-0.5, 0.5, 1000)
    dev = DeviceParams.sample_random("A", seed=7)
    dev.etalon_amplitude = 0.0
    scan = simulate_scan(35.0, 296.0, 1.0, dev, np.random.default_rng(7), lines, include_noise=False)
    fit = levenberg_marquardt_fit(
        scan["voltage_v"], scan["nu_axis_cm1"], t_norm, 296.0, 1.0, dev.path_length_cm, lines, initial_x_ppm=10.0
    )
    diffs = []
    for s in range(n_scipy_scans):
        d = DeviceParams.sample_random("S", seed=100 + s)
        sc = simulate_scan(15.0, 296.0, 1.0, d, np.random.default_rng(s), lines)
        r = levenberg_marquardt_fit(
            sc["voltage_v"], sc["nu_axis_cm1"], t_norm, 296.0, 1.0, d.path_length_cm, lines, initial_x_ppm=5.0
        )

        def residual(theta, sc=sc, d=d):
            alpha = compute_absorption_coefficient(sc["nu_axis_cm1"], theta[0], 296.0, 1.0, lines)
            return np.polyval(theta[1:][::-1], t_norm) * np.exp(-alpha * d.path_length_cm) - sc["voltage_v"]

        ref = least_squares(residual, x0=r["initial_theta"])
        diffs.append(abs(r["x_ppm"] - ref.x[0]))
    return {
        "noiseless_rel_err": float(abs(fit["x_ppm"] - 35.0) / 35.0),
        "noiseless_iterations": int(fit["iterations"]),
        "noiseless_converged": bool(fit["converged"]),
        "vs_scipy_n_scans": n_scipy_scans,
        "vs_scipy_max_abs_diff_ppm": float(max(diffs)) if diffs else None,
        "vs_scipy_median_abs_diff_ppm": float(np.median(diffs)) if diffs else None,
    }


def monte_carlo_checks(lines: list[dict], n_scans: int) -> dict:
    def device(**mods) -> DeviceParams:
        dev = DeviceParams.sample_random("MC", seed=MC_DEVICE_SEED)
        for key, value in mods.items():
            setattr(dev, key, value)
        return dev

    base = device()
    return {
        "device": {
            "white_noise_std_v": base.white_noise_std_v, "flicker_noise_std_v": base.flicker_noise_std_v,
            "etalon_amplitude": base.etalon_amplitude, "drift_rate_v_s": base.drift_rate_v_s,
        },
        "white_noise_only": [run_monte_carlo(x, n_scans, lines, device(**IDEAL_NOISE)) for x in (0.0, 5.0, 25.0, 50.0)],
        "default_noise": [run_monte_carlo(x, n_scans, lines, device()) for x in (5.0, 25.0, 50.0)],
    }


def allan_checks() -> dict:
    white = np.random.default_rng(101).normal(0.0, 1.0, 20000)
    taus, adev = compute_allan_deviation(white, 100.0)
    white_slope = float(np.polyfit(np.log10(taus[:12]), np.log10(adev[:12]), 1)[0])

    series = build_synthetic_series(10000, 100.0, seed=42)
    tau, dev = compute_allan_deviation(series, 100.0)
    i = int(np.argmin(dev))
    return {
        "white_noise_slope_first12": white_slope,
        "fig3_series": {
            "n_samples": 10000, "rate_hz": 100.0,
            "slope_first8": float(np.polyfit(np.log10(tau[:8]), np.log10(dev[:8]), 1)[0]),
            "tau_min_s": float(tau[i]), "adev_min_v": float(dev[i]),
        },
    }


def calibration_checks(lines: list[dict], n_devices: int = 8) -> dict:
    """Max |nu_calibrated - nu_true| of the etalon calibration, with the anchor set to the true nu at sample 500."""
    errors = []
    for seed in range(n_devices):
        dev = DeviceParams.sample_random("c", seed=seed)
        scan = simulate_scan(20.0, 296.0, 1.0, dev, np.random.default_rng(seed), lines)
        nu = calibrate_wavelength_axis(
            scan["time_s"], scan["etalon_channel_v"], dev.etalon_fsr_cm1, float(scan["nu_axis_cm1"][500])
        )
        errors.append(float(np.max(np.abs(nu - scan["nu_axis_cm1"]))))
    return {"n_devices": n_devices, "max_abs_error_cm1_per_device": errors,
            "min_cm1": min(errors), "max_cm1": max(errors)}


def run_validation(lines: list[dict], *, n_mc_scans: int = 1000, n_scipy_scans: int = 30) -> dict:
    return {
        "config": {"n_mc_scans": n_mc_scans, "n_scipy_scans": n_scipy_scans, "mc_device_seed": MC_DEVICE_SEED},
        "lineshape": lineshape_checks(),
        "fit": fit_checks(lines, n_scipy_scans),
        "monte_carlo": monte_carlo_checks(lines, n_mc_scans),
        "allan": allan_checks(),
        "wavelength_calibration": calibration_checks(lines),
    }
