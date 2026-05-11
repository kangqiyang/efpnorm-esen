"""Optimizer setup with vecLR correction.

Vector parameters (vec_proj, vec1_proj, vec2_proj, vec_neighbor,
output_vec_norm) receive systematically weaker gradients under
double-backprop (~5x attenuated). We correct this directly via LR.

vecLR5x is the single largest optimization win: 8-14% across L6/L12/L24.
AK + vecLR are redundant — don't combine them.
"""
import torch.optim as optim

VEC_PARAM_NAMES = {
    'vec_proj', 'vec1_proj', 'vec2_proj', 'vec_neighbor', 'output_vec_norm',
}
EFP_C_PARAM_NAMES = {'log_c_raw'}


def build_optimizer(model, base_lr: float, weight_decay: float,
                    vec_lr_mult: float = 5.0,
                    c_lr_mult: float = 1.0) -> optim.AdamW:
    """Build AdamW with per-group LR for vector and EFPNorm-c parameters."""
    vec_params, c_params, other_params = [], [], []

    for name, param in model.named_parameters():
        short = name.split('.')[-1]
        if short in VEC_PARAM_NAMES:
            vec_params.append(param)
        elif short in EFP_C_PARAM_NAMES:
            c_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
        # vecLR: no weight decay (Rui's finding: WD hurts vector params)
        {'params': vec_params, 'lr': base_lr * vec_lr_mult, 'weight_decay': 0.0},
    ]
    if c_params:
        param_groups.append(
            {'params': c_params, 'lr': base_lr * c_lr_mult, 'weight_decay': 0.0}
        )

    return optim.AdamW(param_groups)


def build_scheduler(optimizer, num_steps: int):
    """Cosine LR decay over full training."""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
