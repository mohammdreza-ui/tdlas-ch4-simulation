#!/usr/bin/env python3
"""Export 300 DPI publication-quality figures and generate standalone interactive HTML report.

Produces:
    docs/figures/fig1_spectroscopy_voigt_fit.png (.svg)
    docs/figures/fig2_physics_vs_ml_comparison.png (.svg)
    docs/figures/fig3_allan_variance_stability.png (.svg)
    docs/figures/fig4_wms_lockin_harmonics.png (.svg)
    docs/interactive_report.html
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdlas.hitran_io import load_hitran_ch4_window
from tdlas.instrument import DeviceParams, simulate_scan
from tdlas.fit import levenberg_marquardt_fit
from tdlas.uncertainty import compute_covariance_and_crb
from tdlas.allan import compute_allan_deviation, build_synthetic_series
from hardware.lockin import software_lockin_detector


def setup_matplotlib_style():
    """Apply clean publication styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.autolayout": True
    })


def export_fig1_spectroscopy(out_dir: Path, lines_data: list[dict]):
    """Figure 1: Transmission, Voigt fit, and Residuals."""
    print("  -> Generating Fig 1: Spectroscopy & Voigt Fit...")
    dev = DeviceParams.sample_random("pub_unit", seed=42)
    dev.etalon_amplitude = 0.005
    dev.white_noise_std_v = 0.0015
    
    rng = np.random.default_rng(123)
    scan = simulate_scan(50.0, 296.0, 1.0, dev, rng, lines_data)
    
    t_norm = (scan["time_s"] - np.mean(scan["time_s"])) / np.ptp(scan["time_s"])
    fit_res = levenberg_marquardt_fit(
        scan["voltage_v"], scan["nu_axis_cm1"], t_norm, 296.0, 1.0, dev.path_length_cm, lines_data
    )
    
    sigma_res = float(np.std(fit_res["residuals"]))
    _, crb_std_x = compute_covariance_and_crb(fit_res["jacobian"], sigma_res)
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1.2]})
    
    # Panel 1: Transmission & Baseline
    axes[0].plot(scan["nu_axis_cm1"], scan["voltage_v"], color="#0284c7", lw=1.2, label=r"Measured Photodiode $V(\nu)$")
    axes[0].plot(scan["nu_axis_cm1"], scan["ground_truth"]["laser_baseline_v"], color="#d97706", lw=1.5, ls="--", label=r"Laser Baseline $P_L(\nu)$")
    axes[0].set_ylabel("Detector Voltage (V)")
    axes[0].set_title(r"(a) Simulated TDLAS scan, 50 ppm $CH_4$, lines at 3017.98 and 3018.42 $\mathrm{cm}^{-1}$")
    axes[0].legend(loc="lower right", framealpha=0.9)
    
    # Panel 2: Absorbance & Voigt Fit
    absorbance_meas = -np.log(np.maximum(scan["voltage_v"] / np.maximum(scan["ground_truth"]["laser_baseline_v"], 1e-4), 1e-4))
    fitted_abs = scan["ground_truth"]["absorbance"] * (fit_res["x_ppm"] / 50.0)
    axes[1].plot(scan["nu_axis_cm1"], absorbance_meas, color="#9333ea", alpha=0.6, lw=1.0, label="Derived Absorbance")
    axes[1].plot(scan["nu_axis_cm1"], fitted_abs, color="#16a34a", lw=2.0, label=rf"True model scaled to fitted $x={fit_res['x_ppm']:.2f}\ \mathrm{{ppm}}$")
    axes[1].set_ylabel(r"Absorbance $A(\nu) = \alpha(\nu) L$")
    axes[1].set_title("(b) Absorbance: from true baseline vs. true model scaled by fitted x / 50")
    axes[1].legend(loc="upper left", framealpha=0.9)
    
    # Panel 3: Residuals
    residuals_mv = fit_res["residuals"] * 1000.0
    axes[2].plot(scan["nu_axis_cm1"], residuals_mv, color="#e11d48", lw=1.0, label="Residuals")
    axes[2].axhline(0, color="gray", lw=1.0, ls="-")
    axes[2].axhline(3.0 * sigma_res * 1000.0, color="#64748b", ls=":", lw=1.0, label=r"$\pm 3\sigma$ of residuals (includes etalon)")
    axes[2].axhline(-3.0 * sigma_res * 1000.0, color="#64748b", ls=":", lw=1.0)
    axes[2].set_xlabel(r"Wavenumber $\nu\ (\mathrm{cm}^{-1})$")
    axes[2].set_ylabel("Error (mV)")
    axes[2].set_title(r"(c) Fit residuals: etalon fringe is not in the fit model")
    axes[2].legend(loc="upper right", framealpha=0.9)
    
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"fig1_spectroscopy_voigt_fit.{ext}", dpi=300)
    plt.close(fig)


ML_RESULTS_PATH = Path(__file__).resolve().parents[1] / "docs" / "results" / "ml_benchmark.json"
SCENARIO_LABELS = {
    "unseen_device": "Unseen\ndevice",
    "ood_temperature_pressure": "T, P out\nof range",
    "higher_noise": "4x noise",
    "stronger_etalon": "Stronger\netalon",
    "axis_error": "Axis\nerror",
    "unknown_interferent": "Unknown\ninterferent",
}


def load_ml_results() -> dict:
    """Load the benchmark written by scripts/run_ml_benchmark.py."""
    if not ML_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{ML_RESULTS_PATH} not found. Run `python scripts/run_ml_benchmark.py` first."
        )
    return json.loads(ML_RESULTS_PATH.read_text(encoding="utf-8"))


def fig2_caption(result: dict) -> str:
    """Caption built from the stored results, so the text cannot drift from the numbers."""
    sc = result["scenarios"]
    n = len(sc)
    better = sum(1 for r in sc.values() if r["classical"]["rmse_ppm"] < r["ml_ensemble"]["rmse_ppm"])
    cov_c = [r["classical"]["coverage_95"] for r in sc.values()]
    cov_e = [r["ml_ensemble"]["coverage_95"] for r in sc.values()]
    cfg = result["config"]
    return (
        f"Results of scripts/run_ml_benchmark.py (docs/results/ml_benchmark.json): classical Levenberg-Marquardt fit vs a 1D CNN and a "
        f"{cfg['n_models']}-member deep ensemble trained on {len(cfg['train_devices'])} simulated devices (validation: {len(cfg['val_devices'])}, "
        f"test: {len(cfg['test_devices'])} held-out devices, {cfg['n_test_scans_per_scenario']} scans per scenario). "
        f"The classical fit has the lower RMSE in {better} of {n} scenarios. Nominal 95% coverage ranges from {min(cov_c):.0%} to {max(cov_c):.0%} "
        f"for the classical fit and from {min(cov_e):.0%} to {max(cov_e):.0%} for the ensemble. The CNN is small, trained on simulated data only, "
        f"with a 3-value learning-rate search, so these numbers do not bound what a better-tuned network could do."
    )


def export_fig2_physics_vs_ml(out_dir: Path, lines_data: list[dict]):
    """Figure 2: classical fit vs CNN / deep ensemble on six scenarios, from the stored benchmark run."""
    print("  -> Generating Fig 2: Physics vs ML Comparison (from docs/results/ml_benchmark.json)...")
    result = load_ml_results()
    names = list(result["scenarios"])
    methods = [("classical", "Classical LM fit", "#0284c7"),
               ("ml_single", "1D CNN (single)", "#f59e0b"),
               ("ml_ensemble", f"Deep ensemble x{result['config']['n_models']}", "#9333ea")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(names))
    w = 0.26

    ax = axes[0, 0]
    for k, (key, label, color) in enumerate(methods):
        vals = [result["scenarios"][n][key]["rmse_ppm"] for n in names]
        lo = [result["scenarios"][n][key]["rmse_ppm"] - result["scenarios"][n][key]["rmse_ci95_ppm"][0] for n in names]
        hi = [result["scenarios"][n][key]["rmse_ci95_ppm"][1] - result["scenarios"][n][key]["rmse_ppm"] for n in names]
        ax.bar(x + (k - 1) * w, vals, w, yerr=[lo, hi], capsize=3, color=color, label=label, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[n] for n in names], fontsize=8)
    ax.set_ylabel("RMSE (ppm), bootstrap 95% CI")
    ax.set_title("(a) Concentration error on held-out devices")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    for k, (key, label, color) in enumerate(methods):
        vals = [result["scenarios"][n][key]["coverage_95"] * 100.0 for n in names]
        ax.bar(x + (k - 1) * w, vals, w, color=color, label=label, alpha=0.9)
    ax.axhline(95.0, color="k", ls="--", lw=1.0, label="nominal 95%")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[n] for n in names], fontsize=8)
    ax.set_ylabel("Empirical coverage of 95% interval (%)")
    ax.set_ylim(0, 125)
    ax.set_title("(b) Interval coverage")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    cmap = plt.get_cmap("tab10")
    for ax, key, title in ((axes[1, 0], "classical", "(c) Reliability: classical fit (CRB intervals)"),
                           (axes[1, 1], "ml_ensemble", "(d) Reliability: deep ensemble")):
        for i, n in enumerate(names):
            rel = result["scenarios"][n]["reliability"]
            ax.plot(rel["levels"], rel[key], "o-", ms=3, lw=1.3, color=cmap(i), label=SCENARIO_LABELS[n].replace("\n", " "))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Nominal central-interval probability")
        ax.set_ylabel("Observed fraction inside")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=7)

    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"fig2_physics_vs_ml_comparison.{ext}", dpi=300)
    plt.close(fig)


def export_fig3_allan_variance(out_dir: Path, is_fast: bool = False):
    """Figure 3: Allan-Werle Deviation Stability."""
    print("  -> Generating Fig 3: Allan-Werle Deviation...")
    n_pts = 10000 if not is_fast else 2000
    fs_hz = 100.0
    y = build_synthetic_series(n_pts, fs_hz, seed=42)
    
    tau_s, adev = compute_allan_deviation(y, fs_hz)
    
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.loglog(tau_s, adev, "o-", color="#0284c7", lw=1.8, ms=4, label=r"Empirical $\sigma_A(\tau)$")
    
    # White noise reference line: slope -0.5
    tau_fit = tau_s[tau_s <= 3.0]
    adev_fit = adev[0] * np.sqrt(tau_s[0] / tau_fit)
    ax.loglog(tau_fit, adev_fit, "--", color="#64748b", lw=1.5, label=r"White Noise ($\tau^{-0.5}$)")
    
    # Minimum point
    min_idx = np.argmin(adev)
    opt_tau = tau_s[min_idx]
    opt_adev = adev[min_idx]
    ax.scatter([opt_tau], [opt_adev], color="#16a34a", s=100, zorder=6, label=rf"$\tau_{{opt}} \approx {opt_tau:.1f}\ \mathrm{{s}}$")
    
    ax.set_xlabel(r"Integration Time $\tau$ (s)")
    ax.set_ylabel(r"Allan Deviation $\sigma_A(\tau)$ (V)")
    ax.set_title("Allan deviation of a synthetic noise series (not fitted concentrations)")
    ax.legend(loc="upper right", framealpha=0.9)
    
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"fig3_allan_variance_stability.{ext}", dpi=300)
    plt.close(fig)


def export_fig4_lockin(out_dir: Path):
    """Figure 4: Wavelength Modulation Harmonic Demodulation (1f and 2f)."""
    print("  -> Generating Fig 4: WMS Harmonic Demodulation...")
    fs_hz = 50000.0
    f_mod_hz = 1000.0
    n_pts = int(fs_hz * 0.02)  # 20 ms
    t = np.arange(n_pts) / fs_hz
    
    carrier = np.sin(2.0 * np.pi * f_mod_hz * t)
    signal = carrier * (1.0 - 0.1 * (1.0 + np.cos(4.0 * np.pi * f_mod_hz * t)))
    noisy = signal + np.random.default_rng(42).normal(0.0, 0.04, n_pts)
    
    r_1f, _ = software_lockin_detector(noisy, f_mod_hz, fs_hz)
    r_2f, _ = software_lockin_detector(noisy, 2.0 * f_mod_hz, fs_hz)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    
    t_ms = t * 1000.0
    ax1.plot(t_ms[:200], noisy[:200], color="#0284c7", lw=1.2, label="Simulated test signal")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("(a) Simulated test signal: carrier with 2f amplitude term (1f and 3f content only)")
    ax1.legend(loc="upper right")
    
    # Demodulated Harmonics Bars
    harmonics = ["Fundamental (1f)", "Second Harmonic (2f)"]
    vals = [r_1f, r_2f]
    colors = ["#0284c7", "#9333ea"]
    
    ax2.bar(harmonics, vals, color=colors, width=0.4, alpha=0.9)
    ax2.set_ylabel("Demodulated RMS (V)")
    ax2.set_title(r"(b) Lock-in outputs: 2f is the noise floor ($2f/1f = " + f"{r_2f/r_1f:.4f}$)")
    
    for ext in ["png", "svg"]:
        fig.savefig(out_dir / f"fig4_wms_lockin_harmonics.{ext}", dpi=300)
    plt.close(fig)


def export_interactive_html(out_dir: Path, figures_dir: Path):
    """Generate standalone interactive HTML report embedding all figures."""
    print("  -> Generating docs/interactive_report.html...")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>TDLAS Methane Spectroscopy — Technical & Metrological Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #080c14; color: #f1f5f9; margin: 0; padding: 24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 24px; color: #06b6d4; margin-bottom: 6px; }}
    .meta {{ color: #94a3b8; font-size: 13px; margin-bottom: 24px; }}
    .fig-card {{ background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
    .fig-card h2 {{ font-size: 16px; margin-top: 0; color: #fff; }}
    .fig-card img {{ width: 100%; max-width: 900px; height: auto; display: block; margin: 12px auto; border-radius: 6px; }}
    .caption {{ font-size: 13px; color: #94a3b8; line-height: 1.6; margin-top: 10px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; background: rgba(6,182,212,0.15); color: #06b6d4; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>TDLAS Mid-IR Methane Spectroscopy Technical Report</h1>
    <div class="meta">
      <span class="badge">4 hand-entered CH4 lines (not verified against HITRAN)</span>
      <span class="badge">3018 cm⁻¹ (3313 nm)</span>
      <span class="badge">Weideman Faddeeva</span>
      <span class="badge">Levenberg-Marquardt</span>
      <span class="badge">Simulated data, assumed 70 cm path</span>
    </div>

    <div class="fig-card">
      <h2>Figure 1: Spectroscopy Transmission, Analytical Voigt Profile, and Residuals</h2>
      <img src="figures/fig1_spectroscopy_voigt_fit.png" alt="Spectroscopy Voigt Fit">
      <div class="caption">
        <strong>Figure 1:</strong> One simulated scan (50 ppm) fitted with the Levenberg-Marquardt code (analytical Jacobian for concentration and baseline polynomial). The etalon fringe is not in the fit model, so the residuals show a periodic pattern of about ±15 mV, and the plotted ±3σ band is computed from these residuals and is therefore much wider than the white noise alone. In panel (b) the "derived absorbance" divides by the true simulated baseline, and the curve labelled fit is the true absorbance scaled by fitted x / 50.
      </div>
    </div>

    <div class="fig-card">
      <h2>Figure 2: Methodological Benchmark — Physics-Informed Inversion vs. Deep Ensembles</h2>
      <img src="figures/fig2_physics_vs_ml_comparison.png" alt="Physics vs ML Comparison">
      <div class="caption">
        <strong>Figure 2:</strong> {fig2_caption(load_ml_results())}
      </div>
    </div>

    <div class="fig-card">
      <h2>Figure 3: Allan-Werle Deviation and Metrological Stability</h2>
      <img src="figures/fig3_allan_variance_stability.png" alt="Allan Variance Stability">
      <div class="caption">
        <strong>Figure 3:</strong> Allan deviation (non-overlapping blocks) of a hand-built synthetic series at 100 Hz: white noise 2 mV, pink noise 0.8 mV rms and a linear drift of 0.3 mV/s. It is not a series of fitted concentrations, and the axis is in volts. The slope over the first points is about -0.46 (the pink noise already bends it), the minimum is 0.41 mV at about 0.9 s, and the rise at longer times comes from the added drift. For pure white noise the same code gives a slope of -0.503.
      </div>
    </div>

    <div class="fig-card">
      <h2>Figure 4: Software Dual-Phase Lock-In Demodulation (1f and 2f WMS)</h2>
      <img src="figures/fig4_wms_lockin_harmonics.png" alt="WMS Lock-In Harmonics">
      <div class="caption">
        <strong>Figure 4:</strong> Software lock-in applied to a simulated test signal: a 1 kHz carrier multiplied by a slowly varying amplitude term containing cos(2·2π·1 kHz·t), plus noise. This signal contains only 1f and 3f components, so the 2f channel shows the noise floor (2f/1f ≈ 0.003), not an absorption signal. It is not a wavelength-modulation spectroscopy simulation, and no concentration is retrieved.
      </div>
    </div>
  </div>
</body>
</html>
"""
    (out_dir / "interactive_report.html").write_text(html_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export publication-quality TDLAS figures.")
    parser.add_argument("--fast", action="store_true", help="Fast generation mode for testing.")
    args = parser.parse_args()

    setup_matplotlib_style()
    
    docs_dir = Path("docs")
    figures_dir = docs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("==================================================")
    print(" Exporting 300 DPI Publication Figures & Reports")
    print(f" Target Directory: {figures_dir.resolve()}")
    print("==================================================")
    
    lines_data = load_hitran_ch4_window(3017.0, 3019.0)
    
    export_fig1_spectroscopy(figures_dir, lines_data)
    export_fig2_physics_vs_ml(figures_dir, lines_data)
    export_fig3_allan_variance(figures_dir, is_fast=args.fast)
    export_fig4_lockin(figures_dir)
    export_interactive_html(docs_dir, figures_dir)
    
    print("==================================================")
    print(" Figures and Interactive Report Export Complete! ")
    print("==================================================")


if __name__ == "__main__":
    main()
