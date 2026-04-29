"""Resume an interrupted stop-gated experiment queue from its manifest.

The Stop 3/4 launchers intentionally run sequentially and may stop at a time
budget boundary.  This helper runs only manifest entries that do not already
have a successful row in ``run_log.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume missing rows from a stop queue manifest")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--cleanup-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest_path = output_root / "manifest.json"
    run_log_path = output_root / "run_log.json"
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(manifest_path, None)
    if manifest is None:
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    run_log: list[dict[str, Any]] = load_json(run_log_path, [])
    successful = {row["config"] for row in run_log if row.get("returncode") == 0}
    missing = [row for row in manifest.get("runs", []) if row["config"] not in successful]
    if args.max_runs is not None:
        missing = missing[: args.max_runs]

    start = time.time()
    print(f"Manifest rows: {len(manifest.get('runs', []))}; successful: {len(successful)}; missing to run: {len(missing)}")
    for row in missing:
        if args.time_budget_hours is not None and (time.time() - start) / 3600.0 >= args.time_budget_hours:
            break
        cfg_path = Path(row["config"])
        cfg = load_json(cfg_path, None)
        if cfg is None:
            raise FileNotFoundError(f"Missing config: {cfg_path}")

        idx = int(row["index"])
        stdout_log = logs_dir / f"{idx:03d}_{cfg['output_dir']}.resume.stdout.log"
        stderr_log = logs_dir / f"{idx:03d}_{cfg['output_dir']}.resume.stderr.log"
        cmd = [sys.executable, "scripts/train.py", "--config", str(cfg_path)]
        elapsed_start = time.time()
        with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
            proc = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f, check=False)
        elapsed = time.time() - elapsed_start

        checkpoint_dir = Path("checkpoints") / cfg["output_dir"]
        checkpoint_removed = False
        if args.cleanup_checkpoints and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            checkpoint_removed = True

        log_row = {
            "index": idx,
            "config": str(cfg_path),
            "output_dir": cfg["output_dir"],
            "dataset": cfg["dataset"],
            "model_type": cfg["model_type"],
            "method": cfg["loss_function"],
            "seed": cfg["seed"],
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "checkpoint_removed": checkpoint_removed,
            "resumed": True,
        }
        if "cost_ratio" in cfg:
            log_row["cost_ratio"] = cfg["cost_ratio"]
        run_log.append(log_row)
        with open(run_log_path, "w") as f:
            json.dump(run_log, f, indent=2)
        if proc.returncode != 0:
            raise RuntimeError(f"Resume run failed: {cfg['output_dir']} (see {stderr_log})")


if __name__ == "__main__":
    main()
