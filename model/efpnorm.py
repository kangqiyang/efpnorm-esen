import torch
import torch.nn as nn
import torch.nn.functional as F


class EFPNorm(nn.Module):
    """Equivariant Force-Preserving Normalization.

    Replaces 1/s with 1/sqrt(s^2 + c^2), keeping the radial singular value
    strictly positive. This makes the Jacobian full-rank so force gradients
    are not destroyed through the autograd chain.

    The learnable softness parameter c (via softplus) controls where
    normalization "turns on":
      - s >> c: behaves like RMSNorm (1/s)
      - s << c: behaves like identity (1/c), preserving magnitude
    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # init c = softplus(0.5413) ~ 1.0
        self.log_c_raw = nn.Parameter(torch.tensor(0.5413))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = F.softplus(self.log_c_raw)
        rms_sq = (x * x).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(rms_sq + c * c)
        return x * scale * self.weight
