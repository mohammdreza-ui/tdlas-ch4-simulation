"""1D Convolutional Neural Network with Gaussian Negative Log-Likelihood and Deep Ensembles."""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

def gaussian_nll_loss(
    mu: torch.Tensor, log_var: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """Heteroscedastic Gaussian Negative Log-Likelihood Loss.

    Formula:
        Loss = 0.5 * exp(-log_var) * (y_true - mu)^2 + 0.5 * log_var
    """
    precision = torch.exp(-log_var)
    squared_diff = (y_true - mu) ** 2
    loss = 0.5 * (precision * squared_diff + log_var)
    return torch.mean(loss)

class TDLAS_Conv1D_Net(nn.Module):
    """1D CNN for absorption spectrum inversion and uncertainty estimation."""
    def __init__(self, input_length: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.head_mu = nn.Linear(128, 1)
        self.head_log_var = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        h = self.fc(feat_flat)
        mu = self.head_mu(h)
        # Clamped log_var for numerical stability
        log_var = torch.clamp(self.head_log_var(h), min=-10.0, max=10.0)
        return mu, log_var

class DeepEnsembleTDLAS:
    """Ensemble of TDLAS networks for epistemic and aleatoric uncertainty.

    The networks are trained on targets divided by `y_scale` (so the NLL is well conditioned);
    `predict` returns the mean and total standard deviation back in the original units (ppm).
    """
    def __init__(self, n_models: int = 5, input_length: int = 1000, y_scale: float = 1.0,
                 models: list[nn.Module] | None = None):
        self.models = models if models is not None else [TDLAS_Conv1D_Net(input_length) for _ in range(n_models)]
        self.y_scale = float(y_scale)

    def member_predict(self, x: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and aleatoric std of a single member, in original units."""
        model = self.models[index]
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            mu, log_var = model(x.to(device))
        return mu * self.y_scale, torch.exp(0.5 * log_var) * self.y_scale

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return combined mean and total standard deviation: sqrt(sigma_aleatoric^2 + sigma_epistemic^2)."""
        means = []
        vars_aleatoric = []
        for model in self.models:
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                mu, log_var = model(x.to(device))
                means.append(mu)
                vars_aleatoric.append(torch.exp(log_var))

        means_tensor = torch.stack(means, dim=0)  # [M, B, 1]
        aleatoric_tensor = torch.stack(vars_aleatoric, dim=0)

        ensemble_mean = torch.mean(means_tensor, dim=0)
        if len(self.models) > 1:
            epistemic_var = torch.var(means_tensor, dim=0, unbiased=True)
        else:
            epistemic_var = torch.zeros_like(ensemble_mean)
        aleatoric_var = torch.mean(aleatoric_tensor, dim=0)

        total_std = torch.sqrt(epistemic_var + aleatoric_var + 1e-12)
        return ensemble_mean * self.y_scale, total_std * self.y_scale


def save_ensemble(ensemble: DeepEnsembleTDLAS, path, meta: dict | None = None) -> None:
    """Save the member weights (CPU tensors), the target scale and optional metadata to one file."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "y_scale": ensemble.y_scale,
        "state_dicts": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in ensemble.models],
        "meta": meta or {},
    }, path)


def load_ensemble(path, device: str = "cpu") -> tuple[DeepEnsembleTDLAS, dict]:
    """Load an ensemble saved by `save_ensemble`; returns (ensemble in eval mode, metadata)."""
    blob = torch.load(path, map_location=device, weights_only=True)
    models = []
    for state in blob["state_dicts"]:
        model = TDLAS_Conv1D_Net()
        model.load_state_dict(state)
        models.append(model.to(device).eval())
    return DeepEnsembleTDLAS(y_scale=blob["y_scale"], models=models), blob.get("meta", {})
