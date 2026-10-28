"""Benchmark of the classical Levenberg-Marquardt fit against a trained 1D CNN and a 5-model deep ensemble.

Protocol
    1. Simulate a dataset of many devices and split it by device into train / validation / test
       (no device appears in two splits).
    2. Tune the learning rate and pick checkpoints on the validation devices only.
    3. Train the ensemble on the training devices.
    4. Evaluate once on six scenarios built from the held-out test devices. The classical fit is
       not tuned on any of this data.

Both methods see the same simulated scans. The classical fit is given the true T and P (a real
sensor measures them), the true path length and, except in the "axis_error" scenario, the true
wavenumber axis. The CNN sees only the normalised voltage trace.
"""
from __future__ import annotations
import dataclasses
import time
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from tdlas.instrument import DeviceParams, simulate_scan
from tdlas.fit import levenberg_marquardt_fit
from tdlas.uncertainty import compute_covariance_and_crb, verify_confidence_coverage
from tdlas.ml.dataset import (
    generate_benchmark_dataset, partition_by_device, normalize_scan,
    TRAIN_T_RANGE_K, TRAIN_P_RANGE_ATM,
)
from tdlas.ml.model import save_ensemble
from tdlas.ml.train import select_learning_rate, train_ensemble, _to_tensors, Y_SCALE_PPM

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "data" / "ensemble_ch4.pt"

TEST_X_RANGE_PPM = (5.0, 45.0)
GROSS_ERROR_PPM = 5.0
RELIABILITY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Placeholder interfering absorber for the "unknown_interferent" scenario. These lines are NOT real
# water-vapour (or any other) HITRAN data: positions and strengths are assumed, chosen to sit between
# the CH4 lines inside the scan window.
INTERFERENT_LINES = [
    {"nu_cm1": 3017.72, "S_T0": 1.0e-19, "gamma_air": 0.070, "gamma_self": 0.30, "n_air": 0.70,
     "delta_air": -0.005, "elower": 100.0},
    {"nu_cm1": 3018.20, "S_T0": 7.0e-20, "gamma_air": 0.070, "gamma_self": 0.30, "n_air": 0.70,
     "delta_air": -0.005, "elower": 100.0},
    {"nu_cm1": 3018.62, "S_T0": 9.0e-20, "gamma_air": 0.070, "gamma_self": 0.30, "n_air": 0.70,
     "delta_air": -0.005, "elower": 100.0},
]

SCENARIO_DESCRIPTIONS = {
    "unseen_device": "Held-out test devices, operating conditions inside the training ranges.",
    "ood_temperature_pressure": "Test devices at T = 330-350 K and P = 1.2-1.4 atm (training: 285-310 K, 0.95-1.05 atm).",
    "higher_noise": "Test devices with white and 1/f noise multiplied by 4.",
    "stronger_etalon": "Test devices with etalon amplitude 4-6% (training: 0.3-1.5%).",
    "axis_error": "True tuning axis differs from the nominal one by up to 0.03 cm^-1 (centre) and 5% (span); the classical fit uses the nominal axis.",
    "unknown_interferent": "Extra assumed absorber (3 placeholder lines, 10-30 ppm equivalent) that neither the fit model nor the training data contain.",
}


def nominal_axis(dev: DeviceParams) -> np.ndarray:
    norm_t = np.linspace(-0.5, 0.5, dev.n_points)
    return dev.nu_center_cm1 + dev.tuning_linear_cm1 * norm_t + dev.tuning_quad_cm1 * norm_t ** 2


def _make_scan(name: str, dev: DeviceParams, rng: np.random.Generator, lines: list[dict]) -> dict:
    """Simulate one test scan for a scenario. Returns the voltage and what the classical fit is told."""
    x = float(rng.uniform(*TEST_X_RANGE_PPM))
    T = float(rng.uniform(*TRAIN_T_RANGE_K))
    P = float(rng.uniform(*TRAIN_P_RANGE_ATM))
    fit_axis = None
    extra = {}
    true_dev = dev
    if name == "ood_temperature_pressure":
        T = float(rng.uniform(330.0, 350.0))
        P = float(rng.uniform(1.2, 1.4))
    elif name == "higher_noise":
        true_dev = dataclasses.replace(
            dev, white_noise_std_v=dev.white_noise_std_v * 4.0, flicker_noise_std_v=dev.flicker_noise_std_v * 4.0
        )
    elif name == "stronger_etalon":
        true_dev = dataclasses.replace(dev, etalon_amplitude=float(rng.uniform(0.04, 0.06)))
    elif name == "axis_error":
        true_dev = dataclasses.replace(
            dev,
            nu_center_cm1=dev.nu_center_cm1 + float(rng.uniform(-0.03, 0.03)),
            tuning_linear_cm1=dev.tuning_linear_cm1 * (1.0 + float(rng.uniform(-0.05, 0.05))),
        )
        fit_axis = nominal_axis(dev)
    elif name == "unknown_interferent":
        extra = {"extra_lines": INTERFERENT_LINES, "extra_x_ppm": float(rng.uniform(10.0, 30.0))}
    elif name != "unseen_device":
        raise ValueError(f"unknown scenario {name}")

    scan = simulate_scan(x, T, P, true_dev, rng, lines, include_noise=True, **extra)
    return {
        "v": scan["voltage_v"], "x": x, "T": T, "P": P, "L": dev.path_length_cm,
        "fit_axis": scan["nu_axis_cm1"] if fit_axis is None else fit_axis,
    }


def _summarize(x_true: np.ndarray, x_hat: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> dict:
    """RMSE (with bootstrap CI), bias, 95% coverage, Gaussian NLL and gross-error rate."""
    finite = np.isfinite(x_hat) & np.isfinite(sigma) & (sigma > 0)
    err = x_hat - x_true
    e, s, xt, xh = err[finite], sigma[finite], x_true[finite], x_hat[finite]
    boot = np.array([
        np.sqrt(np.mean(e[rng.integers(0, len(e), len(e))] ** 2)) for _ in range(500)
    ]) if len(e) else np.array([np.nan])
    s_safe = np.maximum(s, 1e-3)
    nll = float(np.mean(0.5 * np.log(2.0 * np.pi * s_safe ** 2) + 0.5 * (e / s_safe) ** 2)) if len(e) else float("nan")
    return {
        "rmse_ppm": float(np.sqrt(np.mean(e ** 2))),
        "rmse_ci95_ppm": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "bias_ppm": float(np.mean(e)),
        "coverage_95": float(verify_confidence_coverage(xh, xt, s_safe)),
        "nll": nll,
        "mean_sigma_ppm": float(np.mean(s)),
        "gross_error_rate": float(np.mean(~finite | (np.abs(err) > GROSS_ERROR_PPM))),
    }


def _reliability(x_true: np.ndarray, x_hat: np.ndarray, sigma: np.ndarray) -> list[float]:
    """Empirical coverage of the central Gaussian interval at each nominal level."""
    finite = np.isfinite(x_hat) & np.isfinite(sigma) & (sigma > 0)
    ratio = np.abs(x_hat[finite] - x_true[finite]) / np.maximum(sigma[finite], 1e-3)
    return [float(np.mean(ratio <= stats.norm.ppf(0.5 + p / 2.0))) for p in RELIABILITY_LEVELS]


def _ml_predict(ensemble, X: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu_e, std_e = ensemble.predict(Xt)
    mu_1, std_1 = ensemble.member_predict(Xt, 0)
    return tuple(a.squeeze(1).cpu().numpy().astype(np.float64) for a in (mu_e, std_e, mu_1, std_1))


def _time_ensemble_cpu(ensemble, X: np.ndarray, n: int = 50) -> dict:
    """Median batch-1 latency on the CPU for one member and for the whole ensemble."""
    import copy
    cpu = copy.deepcopy(ensemble)
    for m in cpu.models:
        m.to("cpu")
    n = min(n, len(X))
    one, five = [], []
    with torch.no_grad():
        cpu.predict(torch.as_tensor(X[:1]))  # warm-up
        for i in range(n):
            xi = torch.as_tensor(X[i:i + 1])
            t0 = time.perf_counter(); cpu.member_predict(xi, 0); one.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); cpu.predict(xi); five.append(time.perf_counter() - t0)
    return {"ml_single_cpu_ms_median": float(np.median(one) * 1e3),
            "ml_ensemble_cpu_ms_median": float(np.median(five) * 1e3)}


def run_benchmark(
    lines_data: list[dict],
    *,
    n_devices: int = 100,
    scans_per_device: int = 100,
    n_test_scans: int = 400,
    epochs: int = 100,
    patience: int = 15,
    lr_grid: tuple[float, ...] = (3e-3, 1e-3, 3e-4),
    n_models: int = 5,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
    weights_path: Path | str | None = None,
) -> dict:
    """Run the full protocol described in the module docstring and return a JSON-serialisable dict."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(seed)

    # 1. Data and device-level split
    dataset = generate_benchmark_dataset(n_devices, scans_per_device, lines_data, seed=seed)
    train, val, test = partition_by_device(dataset, train_ratio=0.6, val_ratio=0.2)
    train_devs, val_devs, test_devs = (sorted(set(s["device_ids"])) for s in (train, val, test))
    assert not (set(train_devs) & set(val_devs)) and not (set(train_devs) & set(test_devs)) \
        and not (set(val_devs) & set(test_devs)), "device leakage between splits"
    log(f"[data] {len(train['y'])} train / {len(val['y'])} val scans on "
        f"{len(train_devs)}/{len(val_devs)}/{len(test_devs)} train/val/test devices")

    # 2. Hyper-parameter choice on validation only, then ensemble training
    kw = dict(epochs=epochs, patience=patience, device=device)
    best_lr, lr_scores = select_learning_rate(
        train["X"], train["y"], val["X"], val["y"], list(lr_grid), seed=seed, **kw
    )
    log(f"[tune] validation NLL per lr: {lr_scores} -> lr={best_lr:g}")
    ensemble, infos = train_ensemble(
        train["X"], train["y"], val["X"], val["y"], n_models=n_models, lr=best_lr, base_seed=seed, **kw
    )
    if weights_path is not None:
        save_ensemble(ensemble, weights_path, meta={
            "source": "scripts/run_ml_benchmark.py", "seed": seed, "n_devices": n_devices,
            "scans_per_device": scans_per_device, "best_lr": best_lr, "n_models": n_models,
            "train_ranges": {"x_ppm": [0.0, 50.0], "T_K": list(TRAIN_T_RANGE_K), "P_atm": list(TRAIN_P_RANGE_ATM)},
        })
        log(f"[save] ensemble weights -> {weights_path}")
    mu_v, std_v, _, _ = _ml_predict(ensemble, val["X"], device)
    val_summary = _summarize(val["y"].astype(np.float64), mu_v, std_v, rng)
    log(f"[train] ensemble of {n_models}: val RMSE {val_summary['rmse_ppm']:.3f} ppm, "
        f"val coverage {val_summary['coverage_95']:.3f}")

    # 3. Scenarios, evaluated once on the held-out test devices
    test_dev_objs = [test["devices"][d] for d in test_devs]
    results: dict = {}
    last_X = None
    for s_idx, name in enumerate(SCENARIO_DESCRIPTIONS):
        srng = np.random.default_rng(seed * 1000 + s_idx)
        V, xs, xc, sc, conv, tms = [], [], [], [], [], []
        for i in range(n_test_scans):
            dev = test_dev_objs[i % len(test_dev_objs)]
            item = _make_scan(name, dev, srng, lines_data)
            t_norm = np.linspace(-0.5, 0.5, len(item["v"]))
            t0 = time.perf_counter()
            fit = levenberg_marquardt_fit(
                item["v"], item["fit_axis"], t_norm, item["T"], item["P"], item["L"], lines_data
            )
            tms.append((time.perf_counter() - t0) * 1e3)
            _, crb = compute_covariance_and_crb(fit["jacobian"], max(float(np.std(fit["residuals"])), 1e-4))
            V.append(item["v"]); xs.append(item["x"]); xc.append(fit["x_ppm"]); sc.append(crb)
            conv.append(bool(fit["converged"]))
        x_true = np.array(xs)
        X = normalize_scan(np.array(V))
        last_X = X
        mu_e, std_e, mu_1, std_1 = _ml_predict(ensemble, X, device)
        classical = _summarize(x_true, np.array(xc), np.array(sc), srng)
        classical["nonconverged_rate"] = float(1.0 - np.mean(conv))
        classical["fit_ms_median"] = float(np.median(tms))
        results[name] = {
            "description": SCENARIO_DESCRIPTIONS[name],
            "n_scans": n_test_scans,
            "classical": classical,
            "ml_single": _summarize(x_true, mu_1, std_1, srng),
            "ml_ensemble": _summarize(x_true, mu_e, std_e, srng),
            "reliability": {
                "levels": RELIABILITY_LEVELS,
                "classical": _reliability(x_true, np.array(xc), np.array(sc)),
                "ml_single": _reliability(x_true, mu_1, std_1),
                "ml_ensemble": _reliability(x_true, mu_e, std_e),
            },
        }
        r = results[name]
        log(f"[test] {name:<26} RMSE classical {r['classical']['rmse_ppm']:.2f} | "
            f"CNN {r['ml_single']['rmse_ppm']:.2f} | ensemble {r['ml_ensemble']['rmse_ppm']:.2f} ppm")

    return {
        "config": {
            "seed": seed, "n_devices": n_devices, "scans_per_device": scans_per_device,
            "n_test_scans_per_scenario": n_test_scans, "n_models": n_models, "epochs_max": epochs,
            "patience": patience, "lr_grid": list(lr_grid), "device": device,
            "train_devices": train_devs, "val_devices": val_devs, "test_devices": test_devs,
            "train_ranges": {"x_ppm": [0.0, 50.0], "T_K": list(TRAIN_T_RANGE_K), "P_atm": list(TRAIN_P_RANGE_ATM)},
            "test_x_range_ppm": list(TEST_X_RANGE_PPM),
        },
        "training": {
            "best_lr": best_lr, "lr_val_nll": lr_scores,
            "members": [{k: v for k, v in i.items() if k != "history"} for i in infos],
            "validation_ensemble": val_summary,
        },
        "scenarios": results,
        "runtime": _time_ensemble_cpu(ensemble, last_X),
    }
