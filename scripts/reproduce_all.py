"""Master reproduction script: runs simulation, fitting, Allan deviation, and ML benchmark."""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from tdlas.hitran_io import load_hitran_ch4_window
from tdlas.instrument import DeviceParams, simulate_scan
from tdlas.fit import levenberg_marquardt_fit
from tdlas.allan import compute_allan_deviation
from tdlas.uncertainty import compute_covariance_and_crb
from tdlas.ml.benchmark import run_benchmark, DEFAULT_WEIGHTS_PATH

def main():
    parser = argparse.ArgumentParser(description="Reproduce all TDLAS pipeline results")
    parser.add_argument("--fast", action="store_true", help="Run minimal iterations for testing")
    args = parser.parse_args()
    
    print("=================================================================")
    print(">>> 1. Loading offline HITRAN line window at 3018 cm^-1...")
    lines = load_hitran_ch4_window(3017.0, 3019.0)
    print(f"Loaded {len(lines)} CH4 lines in 3017 - 3019 cm^-1 window.")
    for l in lines:
        print(f"  Line: {l['nu_cm1']:.3f} cm^-1 | S(T0): {l['S_T0']:.2e} | gamma_air: {l['gamma_air']}")
        
    print("\n>>> 2. Simulating sensor scan (25.0 ppm, 296 K, 1 atm, L=70 cm)...")
    dev = DeviceParams.sample_random("REPRO_DEV", seed=42)
    rng = np.random.default_rng(42)
    scan = simulate_scan(x_ppm=25.0, T_K=296.0, P_atm=1.0, device=dev, rng=rng, lines_data=lines)
    print(f"Scan duration: {dev.scan_time_s*1000:.1f} ms | Points: {len(scan['voltage_v'])} | ADC: {dev.adc_bits}-bit")
    
    print("\n>>> 3. Executing Levenberg-Marquardt fit from scratch...")
    t_norm = np.linspace(-0.5, 0.5, len(scan["voltage_v"]))
    fit_res = levenberg_marquardt_fit(
        scan["voltage_v"], scan["nu_axis_cm1"], t_norm, 296.0, 1.0, dev.path_length_cm, lines, initial_x_ppm=5.0
    )
    print(f"Retrieved concentration: {fit_res['x_ppm']:.3f} ppm (Ground Truth: 25.000 ppm)")
    print(f"Convergence: {fit_res['converged']} in {fit_res['iterations']} iterations")
    
    # Uncertainty
    sigma_noise = float(np.std(fit_res["residuals"]))
    cov, crb_std = compute_covariance_and_crb(fit_res["jacobian"], max(sigma_noise, 1e-4))
    print(f"Estimated residual noise std: {sigma_noise*1000:.3f} mV")
    print(f"Cramér-Rao Bound (CRB) 1-sigma uncertainty: {crb_std:.3f} ppm")
    
    print("\n>>> 4. Computing Allan-Werle deviation...")
    n_pts = 2000 if args.fast else 10000
    noise_stream = rng.normal(0.0, 0.2, n_pts)
    taus, oadev = compute_allan_deviation(noise_stream, rate_hz=100.0)
    print(f"Computed {len(taus)} Allan points from tau = {taus[0]:.3f}s to {taus[-1]:.3f}s")
    print(f"White noise log-log slope: {np.polyfit(np.log10(taus[:8]), np.log10(oadev[:8]), 1)[0]:.3f} (theoretical: -0.500)")
    
    print("\n>>> 5. Training CNN / deep ensemble and running the 6-scenario benchmark...")
    if args.fast:
        bench = run_benchmark(
            lines, n_devices=10, scans_per_device=12, n_test_scans=12, epochs=3, patience=2,
            lr_grid=(1e-3,), n_models=2, device="cpu", verbose=False,
        )
        print("(--fast: tiny smoke-test configuration, results are meaningless and are not saved)")
    else:
        bench = run_benchmark(lines, verbose=True, weights_path=DEFAULT_WEIGHTS_PATH)
        out = Path("docs/results/ml_benchmark.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bench, indent=2), encoding="utf-8")
        print(f"Saved {out}")
    print("\nBenchmark Evaluation Summary (RMSE in ppm, 95% interval coverage):")
    print(f"{'Scenario':<26} | {'LM RMSE':<8} | {'CNN RMSE':<9} | {'Ens RMSE':<9} | {'LM cov':<7} | {'Ens cov':<7}")
    print("-" * 80)
    for name, m in bench["scenarios"].items():
        print(f"{name:<26} | {m['classical']['rmse_ppm']:<8.2f} | {m['ml_single']['rmse_ppm']:<9.2f} | "
              f"{m['ml_ensemble']['rmse_ppm']:<9.2f} | {m['classical']['coverage_95']:<7.1%} | {m['ml_ensemble']['coverage_95']:<7.1%}")
        
    print("\n=================================================================")
    print(">>> PIPELINE REPRODUCTION COMPLETE <<<")
    print("=================================================================")

if __name__ == "__main__":
    main()
