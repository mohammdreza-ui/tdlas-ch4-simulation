#!/usr/bin/env python3
"""Reproduce the numerical checks quoted in README.md and docs/technical_report.md.

Writes docs/results/validation.json. With --fast a small configuration is run and nothing is written
unless --out is given.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdlas.hitran_io import load_hitran_ch4_window
from tdlas.validation import run_validation

DEFAULT_OUT = Path("docs/results/validation.json")


def print_summary(r: dict) -> None:
    ls, fit, mc, al = r["lineshape"], r["fit"], r["monte_carlo"], r["allan"]
    print(f"Faddeeva max rel. error vs wofz: {ls['faddeeva_max_rel_err_wide_grid']:.2e} (wide grid), "
          f"{ls['faddeeva_max_rel_err_ch4_grid']:.2e} (CH4 grid)")
    print(f"Voigt area (236 cm-1 window): {ls['voigt_area_236cm1_window']:.4f}; Q(296 K) = {ls['partition_function_296K']:.1f}")
    print(f"Noiseless fit rel. error: {fit['noiseless_rel_err']:.2e} in {fit['noiseless_iterations']} iterations")
    print(f"LM vs scipy over {fit['vs_scipy_n_scans']} scans: max {fit['vs_scipy_max_abs_diff_ppm']:.2e} ppm, "
          f"median {fit['vs_scipy_median_abs_diff_ppm']:.2e} ppm")
    print(f"\nMonte Carlo ({r['config']['n_mc_scans']} scans per row)")
    print(f"{'case':<18} {'x':>5} {'mean':>8} {'std':>7} {'CRB':>7} {'CRB/std':>8} {'cov95':>7} {'fail':>6}")
    for case in ("white_noise_only", "default_noise"):
        for row in mc[case]:
            print(f"{case:<18} {row['x_ppm']:5.0f} {row['mean_estimate']:8.3f} {row['std_empirical']:7.3f} "
                  f"{row['mean_crb_std']:7.3f} {row['crb_over_empirical']:8.2f} {row['coverage_95']:7.1%} "
                  f"{row['nonconverged_rate']:6.1%}")
    f3 = al["fig3_series"]
    print(f"\nAllan: white-noise slope {al['white_noise_slope_first12']:.3f}; Fig. 3 series slope(first 8) "
          f"{f3['slope_first8']:.2f}, minimum {f3['adev_min_v'] * 1e3:.2f} mV at tau = {f3['tau_min_s']:.2f} s")
    wc = r["wavelength_calibration"]
    print(f"Etalon axis calibration, max |error| over {wc['n_devices']} devices (true anchor): "
          f"{wc['min_cm1']:.3f}-{wc['max_cm1']:.3f} cm-1")
    d = mc["device"]
    print(f"Monte Carlo device: white {d['white_noise_std_v']*1e3:.2f} mV, 1/f {d['flicker_noise_std_v']*1e3:.2f} mV, "
          f"etalon {d['etalon_amplitude']:.2%}, drift rate {d['drift_rate_v_s']*1e3:.1f} mV/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="small smoke-test configuration")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON result")
    args = parser.parse_args()

    lines = load_hitran_ch4_window(3017.0, 3019.0)
    result = run_validation(lines, **(dict(n_mc_scans=20, n_scipy_scans=3) if args.fast else {}))
    print_summary(result)

    out = args.out if args.out is not None else (None if args.fast else DEFAULT_OUT)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
