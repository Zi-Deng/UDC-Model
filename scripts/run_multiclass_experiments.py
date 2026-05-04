"""Create and execute NICME multiclass experiment matrices.

The runner mirrors the binary Stop1-Stop4 scaffold while keeping multiclass
cost matrices fixed by dataset profile. It writes a plan JSON, per-run configs,
logs, and a CSV ledger suitable for periodic monitoring and paper audit trails.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from nicme.dataset_profiles import get_profile
from nicme.storage_preflight import build_storage_safe_env, run_storage_preflight
from scripts.run_binary_experiments import MODEL_PRESETS

METHODS = (
    "ce",
    "ce_calibrated_cost_min",
    "menon_logit_adjusted",
    "cs_regularized_ce",
    "nicme_logit_adjustment",
    "nicme_hybrid",
)

MODEL_SLUGS = {
    "convnext": "cnx",
    "facebook_dinov3_vit_lora": "fb_d3vit_lora",
    "facebook_dinov3_convnext_lora": "fb_d3cnx_lora",
    "timm_dinov3_vit": "d3vit",
    "timm_dinov3_convnext": "d3cnx",
    "timm_dinov3_vit_lora": "d3vit_lora",
    "timm_dinov3_convnext_lora": "d3cnx_lora",
}

METHOD_SLUGS = {
    "ce": "ce",
    "ce_calibrated_cost_min": "ce_costmin",
    "menon_logit_adjusted": "menon",
    "cs_regularized_ce": "cs_ce",
    "nicme_logit_adjustment": "nicme_logit",
    "nicme_hybrid": "nicme_hybrid",
}

STOP_DEFAULTS = {
    "mc1": {
        "seeds": [42],
        "epochs": 1,
        "models": ["facebook_dinov3_vit_lora", "facebook_dinov3_convnext_lora"],
        # CE calibrated-cost-min is a decision report from the CE run, not a separate training job.
        "methods": ["ce", "nicme_hybrid"],
        "max_train_samples": 320,
        "max_eval_samples": 160,
        "max_test_samples": 160,
    },
    "mc2": {
        "seeds": [42],
        "epochs": 8,
        "models": ["facebook_dinov3_vit_lora", "facebook_dinov3_convnext_lora"],
        "methods": [
            "ce",
            "menon_logit_adjusted",
            "cs_regularized_ce",
            "nicme_logit_adjustment",
            "nicme_hybrid",
        ],
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_test_samples": None,
    },
    "mc3": {
        "seeds": [42, 43, 44, 45, 46],
        "epochs": 20,
        "models": ["facebook_dinov3_vit_lora", "facebook_dinov3_convnext_lora"],
        "methods": ["ce", "cs_regularized_ce", "nicme_logit_adjustment", "nicme_hybrid"],
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_test_samples": None,
    },
    "mc4": {
        "seeds": [42, 43, 44],
        "epochs": 15,
        "models": ["facebook_dinov3_vit_lora", "facebook_dinov3_convnext_lora"],
        "methods": ["ce", "cs_regularized_ce", "nicme_logit_adjustment", "nicme_hybrid"],
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_test_samples": None,
    },
}

LEDGER_FIELDS = [
    "run_id",
    "config_hash",
    "dataset",
    "variant",
    "model_family",
    "method",
    "seed",
    "status",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "config_path",
    "stdout_log",
    "stderr_log",
    "returncode",
    "failure_reason",
    "attempts",
    "attempt_returncodes",
]

GPU_TAINT_LOG_PATTERNS = (
    "cuda error: invalid resource handle",
    "isDifferentiableType(grad.scalar_type()) INTERNAL ASSERT FAILED".lower(),
    "segmentation fault",
    "cuda error: an illegal memory access",
)

RETRYABLE_LOG_PATTERNS = (
    "ValueError: not enough values to unpack (expected 2, got 1)",
    "module.py\", line 2797, in named_children",
)

CUDA_CANARY_CODE = """
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
device = torch.device("cuda:0")
torch.cuda.set_device(device)
x = torch.randn((256, 256), device=device, requires_grad=True)
y = (x @ x.T).mean()
y.backward()
torch.cuda.synchronize(device)
print("cuda_canary_ok")
"""


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or not str(value).strip():
        return default
    return [part.strip() for part in str(value).split(",") if part.strip()]


def tuned_method_settings(dataset: str, method: str) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "loss_function": method,
        "cs_lambda": 0.0,
        "cs_warmup_epochs": 0,
        "nicme_logit_cost_scale": 1.0,
        "logit_adjustment_tau": 1.0,
    }
    if method == "cs_regularized_ce":
        settings["cs_lambda"] = 0.25 if dataset == "eyepacs_dr" else 0.10
        settings["cs_warmup_epochs"] = 1
    elif method == "nicme_logit_adjustment":
        settings["nicme_logit_cost_scale"] = 0.30 if dataset == "eyepacs_dr" else 0.20
    elif method == "nicme_hybrid":
        settings["nicme_logit_cost_scale"] = 0.30 if dataset == "eyepacs_dr" else 0.20
        settings["cs_lambda"] = 0.25 if dataset == "eyepacs_dr" else 0.10
        settings["cs_warmup_epochs"] = 1
    return settings


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "run"


def split_dir_for(dataset: str, variant: str) -> str:
    return f"data/prepared/{dataset}/splits/{variant}"


def output_dir_for(stop: str, dataset: str, variant: str, model_family: str, method: str, seed: int) -> str:
    model_slug = MODEL_SLUGS.get(model_family, model_family)
    method_slug = METHOD_SLUGS.get(method, method)
    return f"{stop}_{dataset}_{variant}_{model_slug}_{method_slug}_s{seed}"


def runtime_config_for(cfg: dict[str, Any], output_dir: Path, attempt: int) -> dict[str, Any]:
    runtime_cfg = dict(cfg)
    namespace = slugify(f"{output_dir.parent.name}_{output_dir.name}")
    storage_key = hashlib.sha256(f"{namespace}:{cfg['output_dir']}:attempt{attempt}".encode()).hexdigest()[:8]
    runtime_cfg["output_dir"] = f"{cfg['output_dir']}_{namespace}_a{attempt}_{storage_key}"
    return runtime_cfg


def make_config(dataset: str, variant: str, model_family: str, method: str, seed: int, stop: str, epochs: int) -> dict:
    profile = get_profile(dataset)
    selection = profile.balanced_selection if variant == "balanced" else profile.natural_selection
    cfg: dict[str, Any] = {
        "dataset_profile": profile.name,
        "variant": variant,
        "dataset": f"{profile.name}_{variant}",
        "dataset_host": "local_folder",
        "local_dataset_format": "manifest",
        "local_folder_path": split_dir_for(profile.name, variant),
        "class_names": list(profile.class_names),
        "num_labels": profile.num_classes,
        "target_recall_class": profile.target_recall_class,
        "target_recall_classes": list(profile.cared_classes),
        "secondary_target_recall_classes": list(profile.secondary_target_recall_classes),
        "target_recall_floor": selection.target_recall_floor,
        "accuracy_floor": selection.accuracy_floor,
        "selection_accuracy_metric": selection.accuracy_metric,
        "cost_matrix": [list(row) for row in profile.cost_matrix],
        "cost_matrix_source": profile.cost_matrix_source,
        "cost_matrix_sha256": profile.cost_matrix_sha256,
        "metric_profile": profile.metric_profile,
        "critical_pairs": [
            {"true": true_name, "pred": pred_name, "cost": cost}
            for true_name, pred_name, cost in profile.critical_pairs
        ],
        "num_train_epochs": epochs,
        "early_stopping_patience": 3,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "wandb": False,
        "push_to_hub": False,
        "calibration_enabled": True,
        "decision_modes": "argmax,calibrated_cost_min",
        "model_selection_metric": "target_recall",
        "save_total_limit": 1,
        "skip_visualizations": stop == "mc1",
        "seed": seed,
        "output_dir": output_dir_for(stop, profile.name, variant, model_family, method, seed),
    }
    cfg.update(MODEL_PRESETS[model_family])
    if model_family == "convnext":
        cfg["disable_cudnn"] = True
    cfg.update(tuned_method_settings(profile.name, method))
    return cfg


def build_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    stop_defaults = STOP_DEFAULTS[args.stop]
    datasets = parse_csv(args.datasets, ["eyepacs_dr", "pmi_pills"])
    variants = parse_csv(args.variants, ["balanced"])
    models = parse_csv(args.models, stop_defaults["models"])
    methods = parse_csv(args.methods, stop_defaults["methods"])
    seeds = [int(seed) for seed in parse_csv(args.seeds, [str(seed) for seed in stop_defaults["seeds"]])]
    epochs = int(args.epochs or stop_defaults["epochs"])

    for dataset in datasets:
        get_profile(dataset)
    for model in models:
        if model not in MODEL_PRESETS:
            raise ValueError(f"Unknown model family {model!r}. Available: {sorted(MODEL_PRESETS)}")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}. Available: {METHODS}")

    runs = []
    for dataset in datasets:
        for variant in variants:
            for model in models:
                for seed in seeds:
                    for method in methods:
                        cfg = make_config(dataset, variant, model, method, seed, args.stop, epochs)
                        for limit_key in ("max_train_samples", "max_eval_samples", "max_test_samples"):
                            value = getattr(args, limit_key) or stop_defaults.get(limit_key)
                            if value is not None:
                                cfg[limit_key] = int(value)
                        runs.append(cfg)
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    return runs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def read_ledger(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["config_hash"]: row for row in csv.DictReader(f)}


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def _log_is_empty(*paths: Path) -> bool:
    return all((not path.exists()) or path.stat().st_size == 0 for path in paths)


def _looks_like_startup_timeout(stdout_log: Path, stderr_log: Path) -> bool:
    if stderr_log.exists() and stderr_log.stat().st_size > 0:
        return False
    if (not stdout_log.exists()) or stdout_log.stat().st_size > 4096:
        return False
    text = stdout_log.read_text(errors="replace")
    return "Using model backend" not in text and "TRAINING COMPLETED" not in text


def _stderr_contains_retryable_failure(stderr_log: Path) -> bool:
    if not stderr_log.exists() or stderr_log.stat().st_size == 0:
        return False
    text = stderr_log.read_text(errors="replace")
    return any(pattern in text for pattern in RETRYABLE_LOG_PATTERNS)


def _is_transient_failure(returncode: int, timed_out: bool, stdout_log: Path, stderr_log: Path) -> bool:
    if returncode == -11:
        return True
    if returncode != 0 and _stderr_contains_retryable_failure(stderr_log):
        return True
    if timed_out and _log_is_empty(stdout_log, stderr_log):
        return True
    if timed_out and _looks_like_startup_timeout(stdout_log, stderr_log):
        return True
    return returncode == 124 and _log_is_empty(stdout_log, stderr_log)


def _read_proc_status(pid_dir: Path) -> dict[str, str]:
    try:
        text = (pid_dir / "status").read_text(errors="replace")
    except OSError:
        return {}
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key] = value.strip()
    return fields


def _d_state_processes() -> list[dict[str, str]]:
    processes = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        status = _read_proc_status(pid_dir)
        state = status.get("State", "")
        if not state.startswith("D"):
            continue
        if status.get("PPid") == "2":
            continue
        processes.append(
            {
                "pid": pid_dir.name,
                "name": status.get("Name", "?"),
                "ppid": status.get("PPid", "?"),
                "state": state,
            }
        )
    return sorted(processes, key=lambda item: int(item["pid"]))


def _gpu_taint_reasons() -> list[str]:
    reasons = []
    d_state = _d_state_processes()
    if d_state:
        sample = ", ".join(f"{item['pid']}:{item['name']}" for item in d_state[:8])
        extra = "" if len(d_state) <= 8 else f", +{len(d_state) - 8} more"
        reasons.append(f"D-state processes present ({sample}{extra})")

    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        reasons.append(f"nvidia-smi probe failed or hung: {exc}")
    else:
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout).strip()
            reasons.append(f"nvidia-smi probe returned {probe.returncode}: {detail}")
    return reasons


def _cuda_canary_reasons(env: dict[str, str], timeout_seconds: float = 45.0, attempts: int = 2) -> list[str]:
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            probe = subprocess.run(
                [sys.executable, "-c", CUDA_CANARY_CODE],
                capture_output=True,
                check=False,
                env=env,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return [f"CUDA canary timed out after {timeout_seconds:.1f}s: {exc}"]
        except OSError as exc:
            return [f"CUDA canary failed to start: {exc}"]
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout).strip()
            failures.append(f"attempt {attempt} returned {probe.returncode}: {detail}")
            continue
        if "cuda_canary_ok" not in probe.stdout:
            failures.append(f"attempt {attempt} returned unexpected output: {probe.stdout.strip()}")
            continue
        return []
    if failures:
        return [f"CUDA canary failed after {attempts} attempts: " + " | ".join(failures)]
    return []


def _stderr_contains_gpu_taint(stderr_log: Path) -> bool:
    if not stderr_log.exists() or stderr_log.stat().st_size == 0:
        return False
    text = stderr_log.read_text(errors="replace").lower()
    return any(pattern in text for pattern in GPU_TAINT_LOG_PATTERNS)


def _terminate_process_group(proc: subprocess.Popen, grace_seconds: float = 15.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill()
        except OSError:
            return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        # A D-state CUDA process cannot be killed from user space. The GPU
        # preflight below will catch it and fail closed before more runs launch.
        return


def _run_train_subprocess(
    config_path: Path,
    stdout_log: Path,
    stderr_log: Path,
    timeout_seconds: float | None,
    env: dict[str, str],
) -> tuple[int, bool]:
    timed_out = False
    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.Popen(
            ["python", "scripts/train.py", "--config", str(config_path)],
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_seconds), timed_out
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc)
            return 124, timed_out


def execute_runs(
    runs: list[dict[str, Any]],
    output_dir: Path,
    resume: bool = True,
    per_run_timeout_minutes: float | None = None,
    retry_transient_once: bool = True,
    require_gpu_preflight: bool = True,
    require_gpu_canary: bool = True,
    gpu_canary_timeout_seconds: float = 45.0,
    abort_on_infrastructure_failure: bool = True,
    abort_on_failure: bool = True,
) -> None:
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "run_ledger.csv"
    existing = read_ledger(ledger_path) if resume else {}
    ledger_rows = list(existing.values())
    training_env = build_storage_safe_env(os.environ)
    storage_report = run_storage_preflight("training", env=training_env, verbose=True)
    write_json(output_dir / "storage_preflight.json", storage_report)

    for index, cfg in enumerate(runs, start=1):
        run_hash = config_hash(cfg)
        if existing.get(run_hash, {}).get("status") == "completed":
            print(f"[{index}/{len(runs)}] Skipping completed run {run_hash}")
            continue
        if require_gpu_preflight:
            taint_reasons = _gpu_taint_reasons()
            if taint_reasons:
                reason = "; ".join(taint_reasons)
                raise SystemExit(f"GPU preflight failed before run {index}/{len(runs)}: {reason}")
        if require_gpu_canary:
            canary_reasons = _cuda_canary_reasons(training_env, timeout_seconds=gpu_canary_timeout_seconds)
            if canary_reasons:
                reason = "; ".join(canary_reasons)
                raise SystemExit(f"CUDA canary failed before run {index}/{len(runs)}: {reason}")

        run_id = f"{index:04d}_{cfg['dataset']}_{cfg['model_type']}_{cfg['loss_function']}_seed{cfg['seed']}"
        attempts_allowed = 2 if retry_transient_once else 1

        start = dt.datetime.now(dt.UTC)
        row = {
            "run_id": run_id,
            "config_hash": run_hash,
            "dataset": cfg["dataset_profile"],
            "variant": cfg["variant"],
            "model_family": cfg["model_type"],
            "method": cfg["loss_function"],
            "seed": cfg["seed"],
            "status": "running",
            "start_time": start.isoformat(),
            "config_path": "",
            "stdout_log": "",
            "stderr_log": "",
            "attempts": 0,
            "attempt_returncodes": "",
        }
        ledger_rows = [old for old in ledger_rows if old.get("config_hash") != run_hash] + [row]
        write_ledger(ledger_path, ledger_rows)
        print(f"[{index}/{len(runs)}] Starting {run_id}")

        attempt_returncodes = []
        failure_reason = ""
        returncode = 1
        for attempt in range(1, attempts_allowed + 1):
            runtime_cfg = runtime_config_for(cfg, output_dir, attempt)
            config_path = config_dir / f"{run_id}.attempt{attempt}.json"
            stdout_log = log_dir / f"{run_id}.attempt{attempt}.stdout.log"
            stderr_log = log_dir / f"{run_id}.attempt{attempt}.stderr.log"
            write_json(config_path, runtime_cfg)
            row["config_path"] = str(config_path)
            row["stdout_log"] = str(stdout_log)
            row["stderr_log"] = str(stderr_log)
            row["attempts"] = attempt
            ledger_rows = [old for old in ledger_rows if old.get("config_hash") != run_hash] + [row]
            write_ledger(ledger_path, ledger_rows)

            timeout_seconds = None if per_run_timeout_minutes is None else float(per_run_timeout_minutes) * 60.0
            returncode, timed_out = _run_train_subprocess(
                config_path,
                stdout_log,
                stderr_log,
                timeout_seconds,
                training_env,
            )
            attempt_returncodes.append(str(returncode))
            if returncode == 0:
                failure_reason = "" if attempt == 1 else f"succeeded on retry attempt {attempt}"
                break

            if timed_out:
                failure_reason = f"train.py timed out after {per_run_timeout_minutes} minutes"
            else:
                failure_reason = f"train.py exited {returncode}"
            if _stderr_contains_gpu_taint(stderr_log):
                failure_reason += "; GPU/CUDA taint signature detected"
                break
            if attempt < attempts_allowed and _is_transient_failure(returncode, timed_out, stdout_log, stderr_log):
                print(f"[{index}/{len(runs)}] retrying transient failure for {run_id}")
                continue
            break
        end = dt.datetime.now(dt.UTC)
        row["end_time"] = end.isoformat()
        row["elapsed_seconds"] = f"{(end - start).total_seconds():.3f}"
        row["returncode"] = returncode
        row["status"] = "completed" if returncode == 0 else "failed"
        row["failure_reason"] = failure_reason
        row["attempt_returncodes"] = ",".join(attempt_returncodes)
        ledger_rows = [old for old in ledger_rows if old.get("config_hash") != run_hash] + [row]
        write_ledger(ledger_path, ledger_rows)
        print(f"[{index}/{len(runs)}] {row['status']} in {row['elapsed_seconds']}s")
        if row["status"] == "failed":
            taint_reasons = _gpu_taint_reasons() if require_gpu_preflight else []
            if abort_on_infrastructure_failure and ("GPU/CUDA taint signature detected" in failure_reason or taint_reasons):
                reason = failure_reason
                if taint_reasons:
                    reason += "; post-run GPU preflight failed: " + "; ".join(taint_reasons)
                raise SystemExit(f"Aborting remaining runs after infrastructure failure in {run_id}: {reason}")
            if abort_on_failure:
                raise SystemExit(f"Aborting remaining runs after failed run {run_id}: {failure_reason}")
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or execute NICME multiclass experiment plans")
    parser.add_argument("--stop", choices=sorted(STOP_DEFAULTS), default="mc1")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset profiles")
    parser.add_argument("--variants", default="balanced", help="Comma-separated prepared split variants")
    parser.add_argument("--models", default=None, help="Comma-separated model families")
    parser.add_argument("--methods", default=None, help="Comma-separated methods")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="results/multiclass")
    parser.add_argument("--per-run-timeout-minutes", type=float, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip conservative D-state/nvidia-smi checks before each training run.",
    )
    parser.add_argument(
        "--skip-gpu-canary",
        action="store_true",
        help="Skip the tiny CUDA forward/backward canary before each training run.",
    )
    parser.add_argument("--gpu-canary-timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--continue-after-infra-failure",
        action="store_true",
        help="Continue the campaign after CUDA taint signatures. Not recommended for paper-grade runs.",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Continue after a failed training run. Not recommended for paper-grade MC2/MC3 campaigns.",
    )
    args = parser.parse_args()

    runs = build_runs(args)
    output_dir = Path(args.output_dir) / args.stop
    write_json(output_dir / "planned_runs.json", {"stop": args.stop, "runs": runs})
    print(f"Wrote {len(runs)} planned runs to {output_dir / 'planned_runs.json'}")
    if args.execute:
        execute_runs(
            runs,
            output_dir,
            resume=not args.no_resume,
            per_run_timeout_minutes=args.per_run_timeout_minutes,
            require_gpu_preflight=not args.skip_gpu_preflight,
            require_gpu_canary=not args.skip_gpu_canary,
            gpu_canary_timeout_seconds=args.gpu_canary_timeout_seconds,
            abort_on_infrastructure_failure=not args.continue_after_infra_failure,
            abort_on_failure=not args.continue_after_failure,
        )


if __name__ == "__main__":
    main()
