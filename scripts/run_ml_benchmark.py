#!/usr/bin/env python3
"""Train the CNN / deep ensemble and run the 6-scenario benchmark against the classical fit.

Writes docs/results/ml_benchmark.json (used for Figure 2) and the trained ensemble weights
src/tdlas/data/ensemble_ch4.pt (used by the dashboard).
With --fast a tiny configuration is run for smoke testing and nothing is written unless --out is given.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdlas.hitran_io import load_hitran_ch4_window
from tdlas.ml.benchmark import run_benchmark, DEFAULT_WEIGHTS_PATH

DEFAULT_OUT = Path("docs/results/ml_benchmark.json")


def fast_kwargs() -> dict:
    return dict(n_devices=10, scans_per_device=12, n_test_scans=12, epochs=3, patience=2,
                lr_grid=(1e-3,), n_models=2, device="cpu")


def print_table(result: dict) -> None:
    print(f"\n{'Scenario':<26} | {'Method':<12} | {'RMSE':>6} | {'Bias':>6} | {'Cov95':>6} | {'NLL':>6} | {'Gross':>6}")
    print("-" * 85)
    for name, r in result["scenarios"].items():
        for key, label in (("classical", "LM fit"), ("ml_single", "CNN"), ("ml_ensemble", "Ensemble x5")):
            m = r[key]
            print(f"{name:<26} | {label:<12} | {m['rmse_ppm']:6.2f} | {m['bias_ppm']:+6.2f} | "
                  f"{m['coverage_95']:6.1%} | {m['nll']:6.2f} | {m['gross_error_rate']:6.1%}")
    rt = result["runtime"]
    print(f"\nCPU batch-1 latency: CNN {rt['ml_single_cpu_ms_median']:.2f} ms, "
          f"ensemble {rt['ml_ensemble_cpu_ms_median']:.2f} ms; "
          f"LM fit median {min(r['classical']['fit_ms_median'] for r in result['scenarios'].values()):.1f}"
          f"-{max(r['classical']['fit_ms_median'] for r in result['scenarios'].values()):.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="tiny smoke-test configuration")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON result")
    args = parser.parse_args()

    lines = load_hitran_ch4_window(3017.0, 3019.0)
    kwargs = fast_kwargs() if args.fast else {"weights_path": DEFAULT_WEIGHTS_PATH}
    result = run_benchmark(lines, **kwargs)
    print_table(result)

    out = args.out if args.out is not None else (None if args.fast else DEFAULT_OUT)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
