"""Lightweight, zero-dependency REST and static asset server for the TDLAS Interactive Dashboard.

Uses Python standard library http.server to provide real-time TDLAS simulation,
Voigt non-linear fitting, Deep Ensemble inference, Allan variance, and lock-in demodulation.
"""

from __future__ import annotations
import json
import socket
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
import numpy as np

from tdlas.hitran_io import load_hitran_ch4_window
from tdlas.instrument import DeviceParams, simulate_scan
from tdlas.fit import levenberg_marquardt_fit
from tdlas.uncertainty import compute_covariance_and_crb
from tdlas.allan import compute_allan_deviation
import sys

# Import lockin and ML (handled gracefully if hardware module is top-level)
try:
    from hardware.lockin import software_lockin_detector
except ImportError:
    # Fallback to local import if sys.path adjusted
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from hardware.lockin import software_lockin_detector

try:
    import torch
    from tdlas.ml.model import load_ensemble
    from tdlas.ml.benchmark import DEFAULT_WEIGHTS_PATH
    from tdlas.ml.dataset import (
        normalize_scan, TRAIN_X_RANGE_PPM, TRAIN_T_RANGE_K, TRAIN_P_RANGE_ATM,
        TRAIN_ETALON_AMPLITUDE_RANGE, TRAIN_WHITE_NOISE_RANGE_V,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_ENSEMBLE_CACHE: dict = {}


def get_trained_ensemble():
    """Load the trained deep ensemble once (CPU). Returns None if PyTorch or the weights file is missing."""
    if not TORCH_AVAILABLE:
        return None
    if "ensemble" not in _ENSEMBLE_CACHE:
        ensemble = None
        if Path(DEFAULT_WEIGHTS_PATH).exists():
            ensemble, _ = load_ensemble(DEFAULT_WEIGHTS_PATH, device="cpu")
            # Warm-up so the reported latency is not dominated by first-call overhead
            ensemble.predict(torch.zeros(1, 1000))
        _ENSEMBLE_CACHE["ensemble"] = ensemble
    return _ENSEMBLE_CACHE["ensemble"]


def training_range_violations(x_ppm: float, T_K: float, P_atm: float, etalon_amp: float, noise_std_v: float) -> list[str]:
    """Which of the requested conditions lie outside the range the ensemble was trained on."""
    checks = [
        ("concentration", x_ppm, TRAIN_X_RANGE_PPM, "ppm"),
        ("temperature", T_K, TRAIN_T_RANGE_K, "K"),
        ("pressure", P_atm, TRAIN_P_RANGE_ATM, "atm"),
        ("etalon amplitude", etalon_amp, TRAIN_ETALON_AMPLITUDE_RANGE, ""),
        ("white noise", noise_std_v, TRAIN_WHITE_NOISE_RANGE_V, "V"),
    ]
    return [f"{name} {value:g} outside {lo:g}-{hi:g} {unit}".strip()
            for name, value, (lo, hi), unit in checks if not lo <= value <= hi]


def ensemble_prediction(voltage: np.ndarray, x_true: float, violations: list[str]) -> dict:
    """Run the trained ensemble on one scan. Nothing here uses the true concentration except the error field."""
    ensemble = get_trained_ensemble()
    if ensemble is None:
        return {"available": False, "x_pred": None, "std_pred": None, "error_ppm": None, "ci95_low": None,
                "ci95_high": None, "latency_ms": None, "in_training_range": None, "out_of_range": [],
                "status": "Trained ensemble not available (needs PyTorch and src/tdlas/data/ensemble_ch4.pt)"}
    x = torch.as_tensor(normalize_scan(voltage), dtype=torch.float32).unsqueeze(0)
    t0 = time.perf_counter()
    mean, std = ensemble.predict(x)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    x_pred, std_pred = float(mean.item()), float(std.item())
    return {
        "available": True,
        "x_pred": x_pred,
        "std_pred": std_pred,
        "error_ppm": abs(x_pred - x_true),
        "ci95_low": x_pred - 1.96 * std_pred,
        "ci95_high": x_pred + 1.96 * std_pred,
        "latency_ms": latency_ms,
        "in_training_range": not violations,
        "out_of_range": violations,
        "status": "Inside training range" if not violations else "Outside training range: " + "; ".join(violations),
    }


PRESETS: dict[str, dict[str, Any]] = {
    "nominal": {
        "name": "Nominal Operating Condition",
        "x_ppm": 50.0,
        "T_K": 296.0,
        "P_atm": 1.0,
        "noise_std_v": 0.0015,
        "etalon_amplitude": 0.005,
        "description": "Standard lab conditions: 50 ppm CH4 at 296 K, 1 atm with minimal optical interference."
    },
    "detection_limit": {
        "name": "Near Detection Limit (LOD)",
        "x_ppm": 1.0,
        "T_K": 296.0,
        "P_atm": 1.0,
        "noise_std_v": 0.0035,
        "etalon_amplitude": 0.002,
        "description": "Ultra-low concentration: 1 ppm CH4 buried in noise; demonstrates CRB and lock-in averaging."
    },
    "etalon_interference": {
        "name": "Severe Optical Fringe Interference",
        "x_ppm": 40.0,
        "T_K": 296.0,
        "P_atm": 1.0,
        "noise_std_v": 0.001,
        "etalon_amplitude": 0.035,
        "description": "Strong parasitic optical cavity fringes (3.5%, above the 1.5% maximum seen in the ensemble's training data)."
    },
    "high_tp": {
        "name": "Extreme Temperature & Pressure Shift",
        "x_ppm": 60.0,
        "T_K": 350.0,
        "P_atm": 1.5,
        "noise_std_v": 0.002,
        "etalon_amplitude": 0.005,
        "description": "Elevated conditions: 350 K, 1.5 atm with collision broadening and spectral pressure shift."
    }
}


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler providing REST endpoints and serving static cockpit files."""
    
    def __init__(self, *args, directory: str | None = None, **kwargs):
        if directory is None:
            static_dir = Path(__file__).parent / "static"
            directory = str(static_dir) if static_dir.exists() else str(Path.cwd())
        super().__init__(*args, directory=directory, **kwargs)

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/presets":
            self._send_json(PRESETS)
        elif self.path.startswith("/api/"):
            self._send_json({"error": "Endpoint not found"}, status=404)
        else:
            # Default to serving static files or index.html
            if self.path == "/" or not self.path:
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            params = json.loads(body)
        except Exception:
            params = {}

        if self.path == "/api/simulate":
            self._handle_simulate(params)
        elif self.path == "/api/fit":
            self._handle_fit(params)
        elif self.path == "/api/compare":
            self._handle_compare(params)
        elif self.path == "/api/allan":
            self._handle_allan(params)
        elif self.path == "/api/lockin":
            self._handle_lockin(params)
        else:
            self._send_json({"error": f"Unknown endpoint {self.path}"}, status=404)

    def _handle_simulate(self, params: dict) -> None:
        x_ppm = float(params.get("x_ppm", 50.0))
        T_K = float(params.get("T_K", 296.0))
        P_atm = float(params.get("P_atm", 1.0))
        noise_std_v = float(params.get("noise_std_v", 0.0015))
        etalon_amp = float(params.get("etalon_amplitude", 0.008))

        dev = DeviceParams.sample_random("web_unit", seed=42)
        dev.white_noise_std_v = noise_std_v
        dev.etalon_amplitude = etalon_amp

        lines_data = load_hitran_ch4_window(3017.0, 3019.0)
        rng = np.random.default_rng(123)
        scan = simulate_scan(x_ppm, T_K, P_atm, dev, rng, lines_data)

        absorbance = scan["ground_truth"]["absorbance"]
        baseline = scan["ground_truth"]["laser_baseline_v"]
        response = {
            "t_ms": (scan["time_s"] * 1000.0).tolist(),
            "nu_axis_cm1": scan["nu_axis_cm1"].tolist(),
            "V_measured": scan["voltage_v"].tolist(),
            "V_noiseless": (baseline * np.exp(-absorbance)).tolist(),
            "baseline": baseline.tolist(),
            "absorbance": absorbance.tolist(),
            "snr_db": float(20.0 * np.log10(max(np.max(absorbance), 1e-6) / max(noise_std_v, 1e-6)))
        }
        self._send_json(response)

    def _handle_fit(self, params: dict) -> None:
        x_ppm = float(params.get("x_ppm", 50.0))
        T_K = float(params.get("T_K", 296.0))
        P_atm = float(params.get("P_atm", 1.0))
        noise_std_v = float(params.get("noise_std_v", 0.0015))
        etalon_amp = float(params.get("etalon_amplitude", 0.008))

        dev = DeviceParams.sample_random("web_unit", seed=42)
        dev.white_noise_std_v = noise_std_v
        dev.etalon_amplitude = etalon_amp

        lines_data = load_hitran_ch4_window(3017.0, 3019.0)
        rng = np.random.default_rng(123)
        scan = simulate_scan(x_ppm, T_K, P_atm, dev, rng, lines_data)

        t_norm = (scan["time_s"] - np.mean(scan["time_s"])) / np.ptp(scan["time_s"])
        
        t0 = time.perf_counter()
        fit_res = levenberg_marquardt_fit(
            scan["voltage_v"], scan["nu_axis_cm1"], t_norm, T_K, P_atm, dev.path_length_cm, lines_data
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        sigma_res = float(np.std(fit_res["residuals"]))
        cov_matrix, crb_std_x = compute_covariance_and_crb(fit_res["jacobian"], sigma_res)

        response = {
            "fitted_x_ppm": fit_res["x_ppm"],
            "true_x_ppm": x_ppm,
            "crb_std_x": crb_std_x,
            "ci95_low": fit_res["x_ppm"] - 1.96 * crb_std_x,
            "ci95_high": fit_res["x_ppm"] + 1.96 * crb_std_x,
            "converged": bool(fit_res["converged"]),
            "iterations": int(fit_res["iterations"]),
            "final_cost": float(fit_res["final_cost"]),
            "residuals": fit_res["residuals"].tolist(),
            "latency_ms": latency_ms,
            "nu_axis_cm1": scan["nu_axis_cm1"].tolist(),
            "V_measured": scan["voltage_v"].tolist()
        }
        self._send_json(response)

    def _handle_compare(self, params: dict) -> None:
        x_ppm = float(params.get("x_ppm", 50.0))
        T_K = float(params.get("T_K", 296.0))
        P_atm = float(params.get("P_atm", 1.0))
        noise_std_v = float(params.get("noise_std_v", 0.0015))
        etalon_amp = float(params.get("etalon_amplitude", 0.008))

        dev = DeviceParams.sample_random("web_unit", seed=42)
        dev.white_noise_std_v = noise_std_v
        dev.etalon_amplitude = etalon_amp

        lines_data = load_hitran_ch4_window(3017.0, 3019.0)
        rng = np.random.default_rng(123)
        scan = simulate_scan(x_ppm, T_K, P_atm, dev, rng, lines_data)

        # 1. Physics Fit (Levenberg-Marquardt)
        t_norm = (scan["time_s"] - np.mean(scan["time_s"])) / np.ptp(scan["time_s"])
        t0 = time.perf_counter()
        fit_res = levenberg_marquardt_fit(
            scan["voltage_v"], scan["nu_axis_cm1"], t_norm, T_K, P_atm, dev.path_length_cm, lines_data
        )
        phys_latency_ms = (time.perf_counter() - t0) * 1000.0
        sigma_res = float(np.std(fit_res["residuals"]))
        _, crb_std_x = compute_covariance_and_crb(fit_res["jacobian"], sigma_res)

        # 2. Trained deep ensemble (weights from scripts/run_ml_benchmark.py), same scan, voltage only
        violations = training_range_violations(x_ppm, T_K, P_atm, etalon_amp, noise_std_v)
        ml = ensemble_prediction(scan["voltage_v"], x_ppm, violations)

        response = {
            "ground_truth_ppm": x_ppm,
            "physics": {
                "x_fitted": fit_res["x_ppm"],
                "crb_std_x": crb_std_x,
                "error_ppm": abs(fit_res["x_ppm"] - x_ppm),
                "ci95_low": fit_res["x_ppm"] - 1.96 * crb_std_x,
                "ci95_high": fit_res["x_ppm"] + 1.96 * crb_std_x,
                "iterations": int(fit_res["iterations"]),
                "latency_ms": phys_latency_ms,
                "converged": bool(fit_res["converged"]),
                "status": "Levenberg-Marquardt fit (CH4 lines, polynomial baseline, no etalon term)"
            },
            "ml": ml
        }
        self._send_json(response)

    def _handle_allan(self, params: dict) -> None:
        n_samples = int(params.get("n_samples", 5000))
        fs_hz = float(params.get("fs_hz", 100.0))
        noise_std_v = float(params.get("noise_std_v", 0.002))

        # Generate realistic photodetector noise time series: white noise + pink noise + drift
        rng = np.random.default_rng(42)
        white = rng.normal(0.0, noise_std_v, n_samples)
        
        # Pink noise
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs_hz)
        freqs[0] = 1e-6
        pink_spectrum = (1.0 / np.sqrt(freqs)) * np.exp(1j * rng.uniform(0, 2*np.pi, len(freqs)))
        pink = np.fft.irfft(pink_spectrum, n=n_samples)
        pink = pink * (0.4 * noise_std_v / (np.std(pink) + 1e-12))
        
        # Drift
        t = np.arange(n_samples) / fs_hz
        drift = 0.0005 * t
        
        y = white + pink + drift
        tau_s, adev = compute_allan_deviation(y, fs_hz)

        # Estimate white noise log-log slope
        log_tau = np.log10(tau_s[: max(3, len(tau_s) // 3)])
        log_adev = np.log10(adev[: max(3, len(adev) // 3)])
        slope, _ = np.polyfit(log_tau, log_adev, 1)

        opt_idx = int(np.argmin(adev))
        opt_tau = float(tau_s[opt_idx])

        response = {
            "tau_s": tau_s.tolist(),
            "adev": adev.tolist(),
            "white_noise_slope": float(slope),
            "optimum_tau_s": opt_tau,
            "min_adev": float(adev[opt_idx])
        }
        self._send_json(response)

    def _handle_lockin(self, params: dict) -> None:
        x_ppm = float(params.get("x_ppm", 40.0))
        T_K = float(params.get("T_K", 296.0))
        P_atm = float(params.get("P_atm", 1.0))
        f_mod_hz = float(params.get("f_mod_hz", 1000.0))
        fs_hz = float(params.get("fs_hz", 50000.0))

        n_pts = int(fs_hz * 0.02)  # 20 ms
        t = np.arange(n_pts) / fs_hz
        
        # Simulated WMS photodiode signal containing absorption modulation and noise
        carrier = np.sin(2.0 * np.pi * f_mod_hz * t)
        absorption_depth = (x_ppm / 100.0) * 0.15
        signal = carrier * (1.0 - absorption_depth * (1.0 + np.cos(4.0 * np.pi * f_mod_hz * t)))
        noisy_signal = signal + np.random.default_rng(42).normal(0.0, 0.05, n_pts)

        r_1f, phase_1f_rad = software_lockin_detector(noisy_signal, f_mod_hz, fs_hz)
        r_2f, phase_2f_rad = software_lockin_detector(noisy_signal, 2.0 * f_mod_hz, fs_hz)
        phase_deg = float(phase_1f_rad * (180.0 / np.pi))

        response = {
            "r_1f": float(r_1f),
            "r_2f": float(r_2f),
            "phase_deg": phase_deg,
            "harmonic_ratio": float(r_2f / max(r_1f, 1e-6)),
            "t_ms": (t[:200] * 1000.0).tolist(),
            "raw_signal": noisy_signal[:200].tolist()
        }
        self._send_json(response)


def start_test_server() -> tuple[HTTPServer, int]:
    """Create and bind an HTTPServer instance to an ephemeral port for testing."""
    server = HTTPServer(("127.0.0.1", 0), DashboardRequestHandler)
    port = server.server_address[1]
    return server, port


def run_server(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Start the interactive dashboard server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    print(f"==================================================")
    print(f" TDLAS Visual Dashboard running at http://{host}:{port}/")
    print(f" Press Ctrl+C to terminate server.")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
        httpd.server_close()
