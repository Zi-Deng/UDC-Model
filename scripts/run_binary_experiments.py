"""Unified binary experiment runner scaffold for NICME paper experiments.

This runner intentionally delegates individual training runs to the existing
training scripts so experiments remain reproducible from JSON configs.  It
creates a manifest of planned runs for Tier 0/1/2/3 schedules and can execute
them sequentially when ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MODEL_PRESETS = {
    "convnext": {
        "model": "facebook/convnext-tiny-224",
        "model_backend": "hf_auto",
        "model_type": "convnext",
        "peft_enabled": False,
    },
    "vit": {
        "model": "google/vit-base-patch16-224-in21k",
        "model_backend": "hf_auto",
        "model_type": "vit",
        "peft_enabled": False,
    },
    "dinov3_convnext": {
        "model": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "model_backend": "dinov3_feature",
        "model_type": "dinov3_convnext",
        "peft_enabled": False,
    },
    "dinov3_vit": {
        "model": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "model_backend": "dinov3_feature",
        "model_type": "dinov3_vit",
        "peft_enabled": True,
        "peft_r": 8,
        "peft_alpha": 16,
        "peft_dropout": 0.1,
        "peft_target_modules": "query,value",
        "peft_modules_to_save": "classifier",
    },
}


TIERS = {
    "tier0": {"seeds": [42], "epochs": 1, "methods": ["ce", "nicme_hybrid"]},
    "tier1": {
        "seeds": [42],
        "epochs": 10,
        "methods": ["ce", "ce_calibrated_cost_min", "menon_logit_adjusted", "nicme_hybrid"],
    },
    "tier2": {
        "seeds": [42, 43, 44, 45, 46],
        "epochs": 20,
        "methods": [
            "ce",
            "ce_calibrated_cost_min",
            "menon_logit_adjusted",
            "cs_regularized_ce",
            "nicme_logit_adjustment",
            "nicme_hybrid",
        ],
    },
    "tier3": {"seeds": [42, 43, 44], "epochs": 15, "methods": ["ce_calibrated_cost_min", "nicme_hybrid"]},
}

TIER_DEFAULT_MODELS = {
    "tier0": ["convnext", "vit", "dinov3_vit"],
    "tier1": ["convnext", "dinov3_vit"],
    "tier2": ["dinov3_vit"],
    "tier3": ["convnext", "vit", "dinov3_convnext", "dinov3_vit"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or execute NICME binary experiment plans")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--tier", choices=sorted(TIERS), default="tier1")
    parser.add_argument(
        "--model-family",
        action="append",
        choices=sorted(MODEL_PRESETS),
        help="Model family to include. Repeat to include multiple. Defaults depend on tier.",
    )
    parser.add_argument("--output", default="results/binary_experiment_plan.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base = json.load(f)

    tier = TIERS[args.tier]
    model_families = args.model_family or TIER_DEFAULT_MODELS[args.tier]
    runs = []
    for model_family in model_families:
        for seed in tier["seeds"]:
            for method in tier["methods"]:
                cfg = dict(base)
                cfg.update(MODEL_PRESETS[model_family])
                cfg["seed"] = seed
                cfg["num_train_epochs"] = tier["epochs"]
                cfg["loss_function"] = method
                cfg["output_dir"] = f"{base.get('output_dir', 'nicme')}_{args.tier}_{model_family}_{method}_seed{seed}"
                runs.append(cfg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"tier": args.tier, "runs": runs}, f, indent=2)
    print(f"Wrote {len(runs)} planned runs to {out}")

    if args.execute:
        temp_dir = out.parent / f"{out.stem}_configs"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for idx, cfg in enumerate(runs):
            cfg_path = temp_dir / f"run_{idx:03d}.json"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            subprocess.run(["python", "scripts/train.py", "--config", str(cfg_path)], check=True)


if __name__ == "__main__":
    main()
