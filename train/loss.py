import torch
import torch.nn.functional as F


def force_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def force_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def energy_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def combined_loss(e_pred, e_target, f_pred, f_target,
                  force_weight: float = 1.0,
                  energy_weight: float = 0.01) -> torch.Tensor:
    """MSE force loss + MAE energy loss (ablations use MSE force only)."""
    f_loss = force_mse(f_pred, f_target)
    e_loss = energy_mae(e_pred, e_target)
    return force_weight * f_loss + energy_weight * e_loss
