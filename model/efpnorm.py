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


class EquivariantEFPNorm(nn.Module):
    """EFPNorm for eSEN's equivariant spherical-harmonic feature arrays.

    Drop-in replacement for fairchem's EquivariantRMSNormArraySphericalHarmonicsV2.
    Swaps the fixed eps denominator for a learnable softplus(log_c_raw)^2 so the
    normalisation scale is bounded away from zero, keeping the Jacobian full-rank
    and preserving force gradients through the norm layers.

    Input shape: [N, (lmax+1)^2, num_channels]
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-5,        # kept for API compat, unused
        affine: bool = True,
        normalization: str = "component",
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # learnable softness: c = softplus(log_c_raw), init ~ 1.0
        self.log_c_raw = nn.Parameter(torch.tensor(0.5413))

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        # precompute the same index/weight buffers as the original
        from fairchem.core.models.uma.nn.layer_norm import get_l_to_all_m_expand_index
        expand_index = get_l_to_all_m_expand_index(lmax)
        self.register_buffer("expand_index", expand_index, persistent=False)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((lmax + 1) ** 2, 1)
            for lval in range(lmax + 1):
                start_idx = lval ** 2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : start_idx + length, :] = 1.0 / length
            balance_degree_weight = balance_degree_weight / (lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight, persistent=False)
        else:
            self.balance_degree_weight = None

    @torch.autocast("cuda", enabled=False)
    @torch.autocast("cpu", enabled=False)
    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        feature = node_input.float()

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0 = feature_l0 - feature_l0.mean(dim=2, keepdim=True)
            feature = torch.cat((feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1)

        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
        else:  # component
            if self.std_balance_degrees:
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature.pow(2), self.balance_degree_weight
                )
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)

        feature_norm = feature_norm.mean(dim=2, keepdim=True)  # [N, 1, 1]

        # EFP: replace (norm + eps)^{-0.5} with (norm + c^2)^{-0.5}
        c = F.softplus(self.log_c_raw)
        feature_norm = (feature_norm + c * c).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(1, self.lmax + 1, self.num_channels)
            weight = torch.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)

        return out
