"""Training of the 1D CNN with Gaussian NLL and of the 5-member deep ensemble.

Only the training and validation splits are used here: the validation split drives early stopping,
checkpoint choice and learning-rate selection. The test devices are never touched in this module.
"""
from __future__ import annotations
import copy
import os
import numpy as np
import torch
from tdlas.ml.model import TDLAS_Conv1D_Net, DeepEnsembleTDLAS, gaussian_nll_loss

Y_SCALE_PPM = 50.0  # targets are trained as y / 50

# Needed for reproducible cuBLAS results; must be set before the first CUDA call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _make_deterministic() -> None:
    """Ask for reproducible GPU kernels (warn only: adaptive_avg_pool backward has no deterministic CUDA kernel)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _to_tensors(X: np.ndarray, y: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y / Y_SCALE_PPM, dtype=torch.float32, device=device).unsqueeze(1)
    return Xt, yt


def evaluate_nll(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Return (mean Gaussian NLL, RMSE in ppm) of a model on a tensor split."""
    model.eval()
    with torch.no_grad():
        mu, log_var = model(X)
        nll = gaussian_nll_loss(mu, log_var, y).item()
        rmse_ppm = float(torch.sqrt(torch.mean((mu - y) ** 2)).item() * Y_SCALE_PPM)
    return nll, rmse_ppm


def train_one_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, lr: float = 1e-3, epochs: int = 100, batch_size: int = 128, patience: int = 15,
    weight_decay: float = 1e-5, seed: int = 0, device: str = "cpu"
) -> tuple[TDLAS_Conv1D_Net, dict]:
    """Train one network with the heteroscedastic Gaussian NLL; keep the best-validation-NLL checkpoint."""
    _make_deterministic()
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = TDLAS_Conv1D_Net(input_length=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt = _to_tensors(X_train, y_train, device)
    Xv, yv = _to_tensors(X_val, y_val, device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_nll = float("inf")
    best_epoch = 0
    history = {"train_nll": [], "val_nll": [], "val_rmse_ppm": []}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=generator).to(device)
        running, n_batches = 0.0, 0
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            mu, log_var = model(Xt[idx])
            loss = gaussian_nll_loss(mu, log_var, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        val_nll, val_rmse = evaluate_nll(model, Xv, yv)
        history["train_nll"].append(running / max(n_batches, 1))
        history["val_nll"].append(val_nll)
        history["val_rmse_ppm"].append(val_rmse)
        if val_nll < best_val_nll:
            best_val_nll, best_epoch = val_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    info = {
        "lr": lr, "seed": seed, "epochs_run": len(history["val_nll"]),
        "best_epoch": best_epoch, "best_val_nll": best_val_nll,
        "best_val_rmse_ppm": history["val_rmse_ppm"][best_epoch - 1],
        "history": history,
    }
    return model, info


def select_learning_rate(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    lr_grid: list[float], **train_kwargs
) -> tuple[float, dict[str, float]]:
    """Pick the learning rate with the lowest validation NLL (single model per value)."""
    scores: dict[str, float] = {}
    for lr in lr_grid:
        _, info = train_one_model(X_train, y_train, X_val, y_val, lr=lr, **train_kwargs)
        scores[f"{lr:g}"] = info["best_val_nll"]
    best = min(lr_grid, key=lambda v: scores[f"{v:g}"])
    return best, scores


def train_ensemble(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    *, n_models: int = 5, lr: float = 1e-3, base_seed: int = 0, **train_kwargs
) -> tuple[DeepEnsembleTDLAS, list[dict]]:
    """Train n_models networks from different seeds (random init and batch order) on the same data."""
    models, infos = [], []
    for k in range(n_models):
        model, info = train_one_model(
            X_train, y_train, X_val, y_val, lr=lr, seed=base_seed + k, **train_kwargs
        )
        models.append(model)
        infos.append(info)
    ensemble = DeepEnsembleTDLAS(y_scale=Y_SCALE_PPM, models=models)
    return ensemble, infos
