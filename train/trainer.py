"""DDP training loop for PolyGET."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .loss import combined_loss, force_mae
from .optimizer import build_optimizer, build_scheduler


def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, rank=0):
        self.rank = rank
        self.cfg = cfg
        self.device = torch.device(f'cuda:{rank}')

        model = model.to(self.device)
        if cfg.get('ddp', False):
            model = DDP(model, device_ids=[rank])
        self.model = model

        self.optimizer = build_optimizer(
            model,
            base_lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 1e-4),
            vec_lr_mult=cfg.get('vec_lr_mult', 5.0),
            c_lr_mult=cfg.get('c_lr_mult', 1.0),
        )
        steps = cfg['epochs'] * len(train_loader)
        self.scheduler = build_scheduler(self.optimizer, steps)

        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            energy, forces = self.model.module.get_forces(
                batch['z'], batch['pos'], batch['edge_index'], batch['batch']
            ) if hasattr(self.model, 'module') else self.model.get_forces(
                batch['z'], batch['pos'], batch['edge_index'], batch['batch']
            )

            loss = combined_loss(
                energy, batch['energy'],
                forces, batch['forces'],
                force_weight=self.cfg.get('force_weight', 1.0),
                energy_weight=self.cfg.get('energy_weight', 0.01),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        all_f_mae = []
        element_errors = {}

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            m = self.model.module if hasattr(self.model, 'module') else self.model
            energy, forces = m.get_forces(
                batch['z'], batch['pos'], batch['edge_index'], batch['batch'])

            mae = (forces - batch['forces']).abs()
            all_f_mae.append(mae.mean().item())

            # Per-element breakdown
            for z_val in batch['z'].unique():
                mask = batch['z'] == z_val
                z_key = z_val.item()
                if z_key not in element_errors:
                    element_errors[z_key] = []
                element_errors[z_key].append(mae[mask].mean().item())

        return {
            'force_mae': sum(all_f_mae) / len(all_f_mae),
            'per_element': {z: sum(v) / len(v) for z, v in element_errors.items()},
        }

    def run(self):
        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch()
            if self.rank == 0:
                metrics = self.validate()
                print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} "
                      f"| val_force_mae={metrics['force_mae']*1000:.1f} meV/Å")
