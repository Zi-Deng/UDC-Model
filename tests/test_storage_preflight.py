from __future__ import annotations

from collections import namedtuple
from pathlib import Path

from nicme.storage_preflight import (
    STORAGE_ENV_KEYS,
    build_storage_report,
    build_storage_safe_env,
    validate_storage_env,
)


def test_build_storage_safe_env_overrides_home_cache_values():
    env = build_storage_safe_env(
        {
            "HF_HOME": str(Path.home() / ".cache" / "huggingface"),
            "TMPDIR": "/tmp",
        }
    )

    for key in STORAGE_ENV_KEYS:
        assert env[key].startswith("/mnt/storage/")
    assert env["HF_HUB_CACHE"] == "/mnt/storage/huggingface/hub"
    assert env["HF_DATASETS_CACHE"] == "/mnt/storage/huggingface/datasets"


def test_validate_storage_env_rejects_home_paths():
    env = build_storage_safe_env({})
    env["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

    errors = validate_storage_env(env)

    assert any("HF_HOME" in error and "home" in error for error in errors)


def test_storage_report_enforces_training_storage_free_space_threshold():
    usage = namedtuple("usage", "total used free")
    gib = 1024**3
    env = build_storage_safe_env({})

    def fake_disk_usage(path: str):
        free_gb = 49 if path == "/mnt/storage" else 20
        return usage(total=200 * gib, used=(200 - free_gb) * gib, free=free_gb * gib)

    report = build_storage_report("training", env, disk_usage=fake_disk_usage)

    assert not report["ok"]
    assert any("storage" in error and "50.00 GB" in error for error in report["errors"])


def test_storage_report_enforces_model_download_storage_free_space_threshold():
    usage = namedtuple("usage", "total used free")
    gib = 1024**3
    env = build_storage_safe_env({})

    def fake_disk_usage(path: str):
        free_gb = 75 if path == "/mnt/storage" else 20
        return usage(total=200 * gib, used=(200 - free_gb) * gib, free=free_gb * gib)

    report = build_storage_report("model_download", env, disk_usage=fake_disk_usage)

    assert not report["ok"]
    assert any("storage" in error and "100.00 GB" in error for error in report["errors"])
