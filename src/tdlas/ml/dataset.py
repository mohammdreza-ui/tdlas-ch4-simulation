"""Leakage-free dataset generation partitioned strictly by hardware device instances."""
from __future__ import annotations
import numpy as np
from tdlas.instrument import DeviceParams, simulate_scan

# Training distribution of the operating conditions (scenarios outside these ranges are "shifted").
TRAIN_X_RANGE_PPM = (0.0, 50.0)
TRAIN_T_RANGE_K = (285.0, 310.0)
TRAIN_P_RANGE_ATM = (0.95, 1.05)
# Ranges of the random instrument parameters in DeviceParams.sample_random
TRAIN_ETALON_AMPLITUDE_RANGE = (0.003, 0.015)
TRAIN_WHITE_NOISE_RANGE_V = (1.0e-3, 2.5e-3)


def normalize_scan(v: np.ndarray) -> np.ndarray:
    """Per-scan normalisation: v / mean(v) - 1, times 10.

    Keeps the relative absorption depth (a z-score would rescale it by the baseline slope).
    Works on a single scan [n] or a batch [B, n].
    """
    v = np.asarray(v, dtype=np.float64)
    mean = np.mean(v, axis=-1, keepdims=True)
    return ((v / np.maximum(mean, 1e-6) - 1.0) * 10.0).astype(np.float32)


def generate_benchmark_dataset(
    num_devices: int,
    scans_per_device: int,
    lines_data: list[dict],
    seed: int = 42
) -> dict:
    """Generate synthetic dataset across multiple device hardware configurations."""
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []
    device_id_list = []
    meta_list = []
    devices: dict[str, DeviceParams] = {}

    for d_idx in range(num_devices):
        dev_id = f"DEV_{d_idx:03d}"
        device = DeviceParams.sample_random(device_id=dev_id, seed=int(rng.integers(1, 1000000)))
        devices[dev_id] = device

        for _ in range(scans_per_device):
            x_ppm = float(rng.uniform(*TRAIN_X_RANGE_PPM))
            T_K = float(rng.uniform(*TRAIN_T_RANGE_K))
            P_atm = float(rng.uniform(*TRAIN_P_RANGE_ATM))

            scan = simulate_scan(x_ppm, T_K, P_atm, device, rng, lines_data, include_noise=True)

            X_list.append(normalize_scan(scan["voltage_v"]))
            y_list.append(x_ppm)
            device_id_list.append(dev_id)
            meta_list.append({"T_K": T_K, "P_atm": P_atm, "device_id": dev_id})

    return {
        "X": np.array(X_list, dtype=np.float32),
        "y": np.array(y_list, dtype=np.float32),
        "device_ids": np.array(device_id_list),
        "metadata": meta_list,
        "devices": devices,
    }

def partition_by_device(
    dataset: dict, train_ratio: float = 0.6, val_ratio: float = 0.2
) -> tuple[dict, dict, dict]:
    """Partition dataset by device ID to prevent leakage of instrument fringe fingerprints."""
    unique_devices = np.unique(dataset["device_ids"])
    n_total = len(unique_devices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_devs = set(unique_devices[:n_train])
    val_devs = set(unique_devices[n_train : n_train + n_val])
    test_devs = set(unique_devices[n_train + n_val :])

    splits = []
    for target_devs in [train_devs, val_devs, test_devs]:
        mask = np.isin(dataset["device_ids"], list(target_devs))
        splits.append({
            "X": dataset["X"][mask],
            "y": dataset["y"][mask],
            "device_ids": dataset["device_ids"][mask],
            "metadata": [m for m, inc in zip(dataset["metadata"], mask) if inc],
            "devices": {k: v for k, v in dataset.get("devices", {}).items() if k in target_devs},
        })

    return splits[0], splits[1], splits[2]
