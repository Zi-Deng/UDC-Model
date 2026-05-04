"""Check storage-safe Hugging Face access for official Facebook DINOv3 models."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from nicme.storage_preflight import build_storage_safe_env, format_storage_report, run_storage_preflight

DEFAULT_DINOV3_MODELS = (
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
)


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_whoami() -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().whoami()
    except Exception as exc:  # noqa: BLE001 - keep auth failures visible in the audit.
        return {"ok": False, "error": repr(exc)}
    return {
        "ok": True,
        "user": info.get("name"),
        "orgs": [org.get("name") for org in info.get("orgs", []) if isinstance(org, dict)],
    }


def _check_model(model_id: str, download_weights: bool) -> dict[str, Any]:
    from transformers import AutoConfig, AutoImageProcessor

    result: dict[str, Any] = {"model_id": model_id, "ok": False, "download_weights": download_weights}
    try:
        config = AutoConfig.from_pretrained(model_id)
        processor = AutoImageProcessor.from_pretrained(model_id)
        result.update(
            {
                "config_model_type": getattr(config, "model_type", None),
                "processor_class": processor.__class__.__name__,
                "hidden_size": getattr(config, "hidden_size", None),
                "hidden_sizes": getattr(config, "hidden_sizes", None),
                "image_size": getattr(config, "image_size", None),
            }
        )
        if download_weights:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_id)
            result["loaded_model_class"] = model.__class__.__name__
            result["parameter_count"] = sum(param.numel() for param in model.parameters())
            del model
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the audit needs the exact access/download failure.
        result["error"] = repr(exc)
    return result


def write_audit(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dinov3_storage_access_audit.json"
    md_path = output_dir / "dinov3_storage_access_audit.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write("# DINOv3 Storage Access Audit\n\n")
        f.write(f"- Timestamp: `{report['timestamp']}`\n")
        f.write(f"- All ready: `{report['all_ready']}`\n")
        f.write(f"- HF user: `{report['huggingface'].get('user', '')}`\n")
        f.write(f"- Download weights: `{report['download_weights']}`\n\n")
        f.write("## Storage Preflight\n\n")
        f.write("```text\n")
        f.write(format_storage_report(report["storage_preflight"]))
        f.write("\n```\n\n")
        f.write("## Models\n\n")
        f.write("| Model | OK | Type | Processor | Parameters | Error |\n")
        f.write("|---|---:|---|---|---:|---|\n")
        for model in report["models"]:
            f.write(
                f"| `{model['model_id']}` | `{model['ok']}` | `{model.get('config_model_type', '')}` | "
                f"`{model.get('processor_class', '')}` | {model.get('parameter_count', '')} | "
                f"{model.get('error', '')} |\n"
            )
    print(f"Wrote DINOv3 storage access audit to {json_path} and {md_path}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    env = build_storage_safe_env(os.environ)
    os.environ.update(env)
    storage_report = run_storage_preflight("model_download", env=env, verbose=True)
    models = [_check_model(model_id, args.download_weights) for model_id in parse_csv(args.models)]
    hf = _safe_whoami()
    all_ready = storage_report["ok"] and hf["ok"] and all(model["ok"] for model in models)
    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "all_ready": all_ready,
        "download_weights": args.download_weights,
        "storage_preflight": storage_report,
        "huggingface": hf,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check storage-safe access to official Facebook DINOv3 models")
    parser.add_argument("--models", default=",".join(DEFAULT_DINOV3_MODELS))
    parser.add_argument("--output-dir", default="results/multiclass/mc_fb0")
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Actually cache full model weights after config/processor access succeeds.",
    )
    args = parser.parse_args()
    report = build_report(args)
    write_audit(report, Path(args.output_dir))
    if not report["all_ready"]:
        raise SystemExit("DINOv3 storage access audit failed; inspect the audit files for details.")


if __name__ == "__main__":
    main()
