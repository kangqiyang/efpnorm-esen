"""
Directly probe the mechanism EFPNorm targets, instead of inferring it from
downstream MD energy drift.

The hypothesis (see model/efpnorm.py, README) is: RMSNorm's scale factor
1/sqrt(s^2 + eps) diverges as the pre-norm activation "energy" s^2 -> 0,
making the norm layer's Jacobian rank-deficient and corrupting the force
gradient (dE/dx) that has to pass back through it. EFPNorm's 1/sqrt(s^2+c^2)
is supposed to stay bounded there instead.

This script runs one forward + conservative-force backward pass (identical
autograd path to training/eval, via torch.autograd.grad) through each of the
9 norm layers, on the same real MD-derived structures already used elsewhere
in this repo, and records at every norm layer, per atom:
  - raw_ms:       the pre-regularizer mean-square activation s^2, computed by
                  replicating the module's internal formula up to (but not
                  including) the "+eps"/"+c^2" step
  - grad_ratio:   ||grad_input|| / ||grad_output|| for that layer during the
                  backward pass -- how much the layer attenuates/amplifies
                  the gradient flowing through it

If the hypothesis is real, RMSNorm should show erratic/blown-up grad_ratio
specifically where raw_ms is small, while EFPNorm should not, on the *same*
structures. If raw_ms never actually gets small in practice, the whole
mechanism is moot regardless of which regularizer is used.

Usage (from efpnorm-esen root):
    python eval/probe_norm_gradients.py --dataset aimnet2 --split val
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.db import connect

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import run_md as rmd  # noqa: E402  (reuses load_checkpoint, atoms_to_atomicdata, eSEN)
from fairchem.core.models.uma.nn.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2
from model.efpnorm import EquivariantEFPNorm

DATA_DIR = _ROOT.parent / "data" / "asedb"
CHECKPOINTS = {
    "efpnorm": _ROOT.parent / "train/checkpoints/aimnet2_L4C128_efpnorm_lr4e-4",
    "rmsnorm": _ROOT.parent / "train/checkpoints/aimnet2_L4C128_rmsnorm_lr4e-4",
}


def raw_ms(module, x: torch.Tensor) -> torch.Tensor:
    """Replicate the module's internal pre-regularizer mean-square (s^2), per atom.

    Identical for both EquivariantRMSNormArraySphericalHarmonicsV2 and
    EquivariantEFPNorm up to the point where the regularizer is added
    (EquivariantEFPNorm is an explicit drop-in replacement of the former).
    """
    feature = x.float()
    if module.centering:
        feature_l0 = feature.narrow(1, 0, 1)
        feature_l0 = feature_l0 - feature_l0.mean(dim=2, keepdim=True)
        feature = torch.cat((feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1)

    if module.normalization == "norm":
        feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
    else:
        if module.std_balance_degrees:
            feature_norm = torch.einsum("nic,ia->nac", feature.pow(2), module.balance_degree_weight)
        else:
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
    feature_norm = feature_norm.mean(dim=2, keepdim=True)  # [N, 1, 1]
    return feature_norm.squeeze(-1).squeeze(-1)  # [N]


def probe_molecule(model, atoms, cutoff, device, norm_layers):
    """Run one forward + conservative-force backward pass with hooks on every
    norm layer. Returns per-layer dict of {raw_ms: [N], grad_ratio: [N]}."""
    data = rmd.atoms_to_atomicdata(atoms, cutoff)
    data = rmd._move_to(data, device)
    data.pos.requires_grad_(True)

    captured = {name: {} for name, _ in norm_layers}
    handles = []

    # Tensor-level hooks (not register_full_backward_hook) because one of the
    # norm layers' forward does an in-place slice-assign on its output
    # (out[:, 0:1, :] = ...), which register_full_backward_hook forbids.
    def make_pre_hook(name, module):
        def hook(mod, inp):
            x = inp[0]
            captured[name]["raw_ms"] = raw_ms(module, x).detach()
            if x.requires_grad:
                def grad_hook(grad):
                    captured[name]["grad_input_norm"] = grad.detach().flatten(1).norm(dim=1)
                x.register_hook(grad_hook)
        return hook

    def make_fwd_hook(name):
        def hook(mod, inp, out):
            if out.requires_grad:
                def grad_hook(grad):
                    captured[name]["grad_output_norm"] = grad.detach().flatten(1).norm(dim=1)
                out.register_hook(grad_hook)
        return hook

    for name, module in norm_layers:
        handles.append(module.register_forward_pre_hook(make_pre_hook(name, module)))
        handles.append(module.register_forward_hook(make_fwd_hook(name)))

    # MLP_EFS_Head computes forces internally via autograd.grad(energy, pos) as
    # part of forward() itself (conservative force path), which already
    # backprops through every norm layer -- that single internal backward is
    # what fires the tensor hooks above. No separate explicit backward needed
    # (and calling one would double-backward through an already-freed graph).
    with torch.enable_grad():
        out = model(data)

    for h in handles:
        h.remove()

    result = {}
    for name, layer_data in captured.items():
        if "raw_ms" in layer_data and "grad_input_norm" in layer_data:
            result[name] = {
                "raw_ms": layer_data["raw_ms"].cpu().numpy(),
                "grad_input_norm": layer_data["grad_input_norm"].cpu().numpy(),
                "grad_output_norm": layer_data["grad_output_norm"].cpu().numpy(),
            }
    return result


def get_norm_layers(model):
    layers = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, (EquivariantRMSNormArraySphericalHarmonicsV2, EquivariantEFPNorm))
    ]
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aimnet2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--indices_from", default=None,
                        help="JSON file whose top-level keys are sample indices to probe "
                             "(default: eval/md_runs/comparison_aimnet2_full200.json)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_json", default=str(_ROOT / "md_runs" / "norm_gradient_probe.json"))
    args = parser.parse_args()

    indices_path = Path(args.indices_from) if args.indices_from else (
        _ROOT / "md_runs" / "comparison_aimnet2_full200.json"
    )
    indices = sorted(json.load(open(indices_path)).keys(), key=int)
    print(f"Probing {len(indices)} molecules from {indices_path.name}")

    db_path = DATA_DIR / f"{args.dataset}_{args.split}.db"
    with connect(str(db_path)) as db:
        rows = list(db.select(limit=max(int(i) for i in indices) + 1))

    summary = {}
    for label, ckpt_dir in CHECKPOINTS.items():
        print(f"\n=== {label} ===")
        model, config = rmd.load_checkpoint(ckpt_dir, "best.pt", args.device)
        cutoff = config["backbone_cfg"]["cutoff"]
        norm_layers = get_norm_layers(model)
        print(f"  {len(norm_layers)} norm layers found")

        regularizer_sq = {}
        for name, module in norm_layers:
            if isinstance(module, EquivariantEFPNorm):
                c = torch.nn.functional.softplus(module.log_c_raw).item()
                regularizer_sq[name] = c * c
            else:
                regularizer_sq[name] = module.eps

        agg = {name: {"raw_ms": [], "grad_input_norm": [], "grad_output_norm": []}
               for name, _ in norm_layers}

        for i, idx in enumerate(indices):
            atoms = rows[int(idx)].toatoms()
            try:
                result = probe_molecule(model, atoms, cutoff, args.device, norm_layers)
            except Exception as ex:
                print(f"  [{idx}] skipped ({ex})")
                continue
            for name, d in result.items():
                agg[name]["raw_ms"].append(d["raw_ms"])
                agg[name]["grad_input_norm"].append(d["grad_input_norm"])
                agg[name]["grad_output_norm"].append(d["grad_output_norm"])
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(indices)} molecules done")

        layer_summary = {}
        for name in agg:
            raw = np.concatenate(agg[name]["raw_ms"])
            gi = np.concatenate(agg[name]["grad_input_norm"])
            go = np.concatenate(agg[name]["grad_output_norm"])
            ratio = gi / (go + 1e-12)
            # split into low/high raw_ms deciles to check for divergence at small s^2
            order = np.argsort(raw)
            n = len(raw)
            lo = order[: max(1, n // 10)]
            hi = order[-max(1, n // 10):]
            reg_sq = regularizer_sq[name]
            layer_summary[name] = {
                "n_atoms": int(n),
                "regularizer_sq": float(reg_sq),
                "raw_ms_min": float(raw.min()),
                "raw_ms_p1": float(np.percentile(raw, 1)),
                "raw_ms_p50": float(np.percentile(raw, 50)),
                "min_raw_ms_over_regularizer_sq": float(raw.min() / reg_sq),
                "grad_ratio_p50": float(np.percentile(ratio, 50)),
                "grad_ratio_p99": float(np.percentile(ratio, 99)),
                "grad_ratio_max": float(ratio.max()),
                "grad_ratio_median_lowest_decile_rawms": float(np.median(ratio[lo])),
                "grad_ratio_median_highest_decile_rawms": float(np.median(ratio[hi])),
            }
            print(f"  {name}: reg_sq={reg_sq:.4g}  raw_ms min={raw.min():.4g} (={raw.min()/reg_sq:.3g}x reg) "
                  f"p1={np.percentile(raw,1):.4g} p50={np.percentile(raw,50):.4g} | "
                  f"grad_ratio p50={np.percentile(ratio,50):.4g} p99={np.percentile(ratio,99):.4g} "
                  f"max={ratio.max():.4g} | low-decile-ratio={np.median(ratio[lo]):.4g} "
                  f"high-decile-ratio={np.median(ratio[hi]):.4g}")

        summary[label] = layer_summary
        del model
        torch.cuda.empty_cache()

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
