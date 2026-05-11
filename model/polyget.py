"""PolyGET: equivariant force field with EFPNorm + vecLR correction."""
import torch
import torch.nn as nn

from .message_passing import PolyGETLayer
from .efpnorm import EFPNorm


class BesselRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        freq = torch.arange(1, num_rbf + 1) * torch.pi / cutoff
        self.register_buffer('freq', freq)
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        envelope = 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1)
        rbf = envelope.unsqueeze(-1) * torch.sin(self.freq * dist.unsqueeze(-1)) / dist.unsqueeze(-1)
        return rbf


class PolyGET(nn.Module):
    """
    L=1 equivariant force field.

    Best config (as of March 17 2026):
      L=24, hidden=128, heads=8, cutoff=5.0 Å,
      EFPNorm, VecNE, vecLR5x (via optimizer), no ResScale, no AK
    Forces computed via autograd: F_i = -dE/dr_i
    """

    def __init__(self,
                 num_layers: int = 24,
                 hidden: int = 128,
                 num_rbf: int = 20,
                 cutoff: float = 5.0,
                 max_z: int = 35,  # Br is heaviest element in ALLOWED_Z
                 use_vec_ne: bool = True):
        super().__init__()

        self.cutoff = cutoff
        self.hidden = hidden

        self.atom_embed = nn.Embedding(max_z + 1, hidden)
        self.rbf = BesselRBF(num_rbf, cutoff)

        self.layers = nn.ModuleList([
            PolyGETLayer(hidden, num_rbf, cutoff, use_vec_ne=use_vec_ne)
            for _ in range(num_layers)
        ])

        self.output_norm = EFPNorm(hidden)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:          [N] atomic numbers
            pos:        [N, 3] atomic positions (requires_grad=True for forces)
            edge_index: [2, E] neighbor list within cutoff
            batch:      [N] batch assignment
        Returns:
            energy:     [B] total energy per structure
        """
        # Edge geometry
        src, dst = edge_index
        diff = pos[dst] - pos[src]                   # [E, 3]
        dist = torch.linalg.norm(diff, dim=-1)        # [E]
        edge_vec = diff / dist.unsqueeze(-1)          # [E, 3] unit vectors
        edge_rbf = self.rbf(dist)                     # [E, num_rbf]

        # Node features
        s = self.atom_embed(z)                        # [N, hidden]
        v = torch.zeros(s.shape[0], self.hidden, 3,
                        device=pos.device)            # [N, hidden, 3]

        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_rbf, edge_vec)

        s = self.output_norm(s)
        e_i = self.energy_head(s).squeeze(-1)         # [N]

        # Sum atomic energies per structure
        B = batch.max().item() + 1
        energy = torch.zeros(B, device=s.device)
        energy.scatter_add_(0, batch, e_i)
        return energy

    def get_forces(self, z, pos, edge_index, batch):
        pos = pos.requires_grad_(True)
        energy = self.forward(z, pos, edge_index, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training)[0]
        return energy, forces
