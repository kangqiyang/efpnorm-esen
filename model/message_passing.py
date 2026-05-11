"""PaiNN-style L=1 equivariant message passing block with EFPNorm."""
import torch
import torch.nn as nn

from .efpnorm import EFPNorm


class MessageBlock(nn.Module):
    """Equivariant message passing: scalar + vector pathways.

    Vector parameters subject to 5x LR multiplier (see optimizer.py):
      vec_proj, vec1_proj, vec2_proj, vec_neighbor
    """

    def __init__(self, hidden: int, num_rbf: int, cutoff: float,
                 use_vec_ne: bool = True):
        super().__init__()
        self.hidden = hidden
        self.use_vec_ne = use_vec_ne

        # Scalar message network
        self.scalar_msg = nn.Sequential(
            nn.Linear(hidden + num_rbf, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * hidden),
        )

        # Vector projections (vecLR parameters)
        self.vec1_proj = nn.Linear(hidden, hidden, bias=False)
        self.vec2_proj = nn.Linear(hidden, hidden, bias=False)
        if use_vec_ne:
            self.vec_neighbor = nn.Linear(3, hidden, bias=False)

    def forward(self, s: torch.Tensor, v: torch.Tensor,
                edge_index: torch.Tensor, edge_rbf: torch.Tensor,
                edge_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:          [N, hidden] scalar features
            v:          [N, hidden, 3] vector features
            edge_index: [2, E] source/target indices
            edge_rbf:   [E, num_rbf] radial basis
            edge_vec:   [E, 3] unit displacement vectors d_ij
        """
        src, dst = edge_index
        N = s.shape[0]

        # Scalar messages
        s_src = s[src]
        msg_in = torch.cat([s_src, edge_rbf], dim=-1)
        msg = self.scalar_msg(msg_in)
        ds, dv_scalar, gate = msg.chunk(3, dim=-1)   # [E, hidden] each

        # Vector messages: combine vec features of source + neighbor directions
        v_src = v[src]                                 # [E, hidden, 3]
        v1 = self.vec1_proj(v_src.transpose(1, 2)).transpose(1, 2)
        v2 = self.vec2_proj(v_src.transpose(1, 2)).transpose(1, 2)

        if self.use_vec_ne and hasattr(self, 'vec_neighbor'):
            v_ne = self.vec_neighbor(edge_vec).unsqueeze(1).expand_as(v1)
            v_msg = v1 + v_ne
        else:
            v_msg = v1

        # Scalar gates modulate vector messages
        v_msg = v_msg * gate.unsqueeze(-1)

        # Aggregate to destination nodes
        ds_agg = torch.zeros(N, self.hidden, device=s.device).scatter_add(
            0, dst.unsqueeze(-1).expand_as(ds), ds)
        dv_agg = torch.zeros(N, self.hidden, 3, device=v.device).scatter_add(
            0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(v_msg), v_msg)

        # Vector contribution from scalar: v2 * scalar_norm
        v_scalar_update = (v2 * dv_scalar.unsqueeze(-1)).sum(dim=0)

        return ds_agg, dv_agg + v_scalar_update


class UpdateBlock(nn.Module):
    """Node-level update: scalar FFN + vector MLP.

    Vector parameters subject to 5x LR multiplier:
      vec_proj, output_vec_norm
    """

    def __init__(self, hidden: int):
        super().__init__()

        # Vector update (vecLR parameters)
        self.vec_proj = nn.Linear(hidden, hidden, bias=False)
        self.output_vec_norm = EFPNorm(hidden)

        # Scalar FFN
        self.scalar_ffn = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * hidden),
        )
        self.scalar_norm = EFPNorm(hidden)

    def forward(self, s: torch.Tensor, v: torch.Tensor,
                ds: torch.Tensor, dv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = v + dv
        vp = self.vec_proj(v.transpose(1, 2)).transpose(1, 2)
        v_norm = torch.linalg.norm(vp, dim=-1)           # [N, hidden]

        ffn_in = torch.cat([s + ds, v_norm], dim=-1)
        ffn_out = self.scalar_ffn(ffn_in)
        a, b, c = ffn_out.chunk(3, dim=-1)

        s = self.scalar_norm(s + ds + a)
        v = self.output_vec_norm(v + vp * b.unsqueeze(-1))

        return s, v


class PolyGETLayer(nn.Module):
    def __init__(self, hidden: int, num_rbf: int, cutoff: float,
                 use_vec_ne: bool = True):
        super().__init__()
        self.message = MessageBlock(hidden, num_rbf, cutoff, use_vec_ne=use_vec_ne)
        self.update = UpdateBlock(hidden)

    def forward(self, s, v, edge_index, edge_rbf, edge_vec):
        ds, dv = self.message(s, v, edge_index, edge_rbf, edge_vec)
        s, v = self.update(s, v, ds, dv)
        return s, v
