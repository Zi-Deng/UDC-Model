"""Storage-safe cache environment and preflight checks for large model runs."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

STORAGE_ROOT = Path("/mnt/storage")

STORAGE_ENV_DEFAULTS = {
    "HF_HOME": STORAGE_ROOT / "huggingface",
    "HF_HUB_CACHE": STORAGE_ROOT / "huggingface" / "hub",
    "HF_DATASETS_CACHE": STORAGE_ROOT / "huggingface" / "datasets",
    "TORCH_HOME": STORAGE_ROOT / "torch",
    "TMPDIR": STORAGE_ROOT / "tmp" / "nicme",
    "XDG_CACHE_HOME": STORAGE_ROOT / ".cache",
    "MPLCONFIGDIR": STORAGE_ROOT / ".cache" / "matplotlib",
}

STORAGE_ENV_KEYS = tuple(STORAGE_ENV_DEFAULTS)
MIN_SYSTEM_FREE_GB = 10.0
MIN_STORAGE_DOWNLOAD_GB = 100.0
MIN_STORAGE_TRAINING_GB = 50.0


class StoragePreflightError(RuntimeError):
    """Raised when a run would risk filling home/root filesystems."""


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_storage_safe_env(
    base_env: dict[str, str] | None = None,
    storage_root: str | Path = STORAGE_ROOT,
) -> dict[str, str]:
    """Return an environment that directs large caches to ``/mnt/storage``."""
    env = dict(os.environ if base_env is None else base_env)
    root = Path(storage_root)
    values = {
        "HF_HOME": root / "huggingface",
        "HF_HUB_CACHE": root / "huggingface" / "hub",
        "HF_DATASETS_CACHE": root / "huggingface" / "datasets",
        "TORCH_HOME": root / "torch",
        "TMPDIR": root / "tmp" / "nicme",
        "XDG_CACHE_HOME": root / ".cache",
        "MPLCONFIGDIR": root / ".cache" / "matplotlib",
    }
    for key, value in values.items():
        env[key] = str(value)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


def ensure_storage_dirs(env: dict[str, str]) -> None:
    """Create cache/temp directories used by storage-safe runs."""
    for key in STORAGE_ENV_KEYS:
        Path(env[key]).mkdir(parents=True, exist_ok=True)


def validate_storage_env(
    env: dict[str, str],
    storage_root: str | Path = STORAGE_ROOT,
    home_path: str | Path | None = None,
    home_root: str | Path = "/home",
) -> list[str]:
    """Return validation errors for cache env values that are unsafe."""
    errors: list[str] = []
    storage = _resolve(storage_root)
    user_home = _resolve(home_path or Path.home())
    home_root_path = _resolve(home_root)
    for key in STORAGE_ENV_KEYS:
        raw = env.get(key)
        if not raw:
            errors.append(f"{key} is not set")
            continue
        value = _resolve(raw)
        if not _is_relative_to(value, storage):
            errors.append(f"{key}={value} is not under {storage}")
        if _is_relative_to(value, user_home) or _is_relative_to(value, home_root_path):
            errors.append(f"{key}={value} resolves under home; use {storage}")
    return errors


def _free_gb(path: Path, disk_usage: Callable[[str], Any]) -> float:
    usage = disk_usage(str(path))
    return float(usage.free) / (1024**3)


def _space_checks(stage: str, disk_usage: Callable[[str], Any]) -> list[dict[str, Any]]:
    storage_threshold = MIN_STORAGE_DOWNLOAD_GB if stage == "model_download" else MIN_STORAGE_TRAINING_GB
    checks = [
        {"name": "root", "path": Path("/"), "min_free_gb": MIN_SYSTEM_FREE_GB},
        {"name": "home_root", "path": Path("/home"), "min_free_gb": MIN_SYSTEM_FREE_GB},
        {"name": "user_home", "path": Path.home(), "min_free_gb": MIN_SYSTEM_FREE_GB},
        {"name": "storage", "path": STORAGE_ROOT, "min_free_gb": storage_threshold},
    ]
    results = []
    for check in checks:
        path = check["path"]
        if not path.exists():
            results.append({**check, "exists": False, "free_gb": None, "ok": False})
            continue
        free = _free_gb(path, disk_usage)
        results.append({**check, "exists": True, "free_gb": free, "ok": free >= check["min_free_gb"]})
    return results


def build_storage_report(
    stage: str,
    env: dict[str, str],
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    """Build a serializable report for storage cache paths and free space."""
    if stage not in {"model_download", "training"}:
        raise ValueError("stage must be 'model_download' or 'training'")
    env_paths = {
        key: {
            "value": env.get(key, ""),
            "resolved": str(_resolve(env[key])) if env.get(key) else "",
        }
        for key in STORAGE_ENV_KEYS
    }
    space = _space_checks(stage, disk_usage)
    errors = validate_storage_env(env)
    for check in space:
        if not check["ok"]:
            if check["exists"]:
                errors.append(
                    f"{check['name']} at {check['path']} has {check['free_gb']:.2f} GB free; "
                    f"requires at least {check['min_free_gb']:.2f} GB"
                )
            else:
                errors.append(f"{check['name']} path {check['path']} does not exist")
    return {
        "stage": stage,
        "storage_root": str(STORAGE_ROOT),
        "env": env_paths,
        "space": [
            {
                "name": item["name"],
                "path": str(item["path"]),
                "exists": item["exists"],
                "free_gb": item["free_gb"],
                "min_free_gb": item["min_free_gb"],
                "ok": item["ok"],
            }
            for item in space
        ],
        "ok": not errors,
        "errors": errors,
    }


def format_storage_report(report: dict[str, Any]) -> str:
    """Format a concise human-readable storage preflight report."""
    lines = [f"Storage preflight ({report['stage']}): ok={report['ok']}"]
    lines.append("Environment:")
    for key in STORAGE_ENV_KEYS:
        item = report["env"][key]
        lines.append(f"  {key}={item['value']}")
    lines.append("Free space:")
    for item in report["space"]:
        free = "missing" if item["free_gb"] is None else f"{item['free_gb']:.2f} GB"
        lines.append(f"  {item['name']} {item['path']}: {free} / required {item['min_free_gb']:.2f} GB")
    if report["errors"]:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in report["errors"])
    return "\n".join(lines)


def run_storage_preflight(
    stage: str,
    env: dict[str, str] | None = None,
    verbose: bool = True,
    disk_usage: Callable[[str], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    """Create dirs, validate cache paths, check free space, and raise on failure."""
    run_env = build_storage_safe_env(env)
    ensure_storage_dirs(run_env)
    report = build_storage_report(stage, run_env, disk_usage=disk_usage)
    if verbose:
        print(format_storage_report(report))
    if not report["ok"]:
        raise StoragePreflightError("; ".join(report["errors"]))
    return report
