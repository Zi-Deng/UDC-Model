import subprocess
from pathlib import Path

from scripts.run_multiclass_experiments import (
    MODEL_SLUGS,
    STOP_DEFAULTS,
    _cuda_canary_reasons,
    _d_state_processes,
    _is_transient_failure,
)


def test_multiclass_defaults_use_official_facebook_dinov3_lora_models():
    expected = ["facebook_dinov3_vit_lora", "facebook_dinov3_convnext_lora"]

    assert STOP_DEFAULTS["mc1"]["models"] == expected
    assert STOP_DEFAULTS["mc2"]["models"] == expected
    assert STOP_DEFAULTS["mc3"]["models"] == expected
    assert STOP_DEFAULTS["mc4"]["models"] == expected
    assert MODEL_SLUGS["facebook_dinov3_vit_lora"] == "fb_d3vit_lora"
    assert MODEL_SLUGS["facebook_dinov3_convnext_lora"] == "fb_d3cnx_lora"


def test_cuda_canary_reports_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    reasons = _cuda_canary_reasons({}, timeout_seconds=1.0)

    assert reasons
    assert "timed out" in reasons[0]


def test_cuda_canary_reports_nonzero_return(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="bad cuda")

    monkeypatch.setattr(subprocess, "run", fake_run)

    reasons = _cuda_canary_reasons({}, timeout_seconds=1.0)

    assert len(reasons) == 1
    assert reasons[0].startswith("CUDA canary failed after 2 attempts")
    assert "bad cuda" in reasons[0]


def test_cuda_canary_recovers_from_transient_nonzero_return(monkeypatch):
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="bad import")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="cuda_canary_ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    reasons = _cuda_canary_reasons({}, timeout_seconds=1.0)

    assert reasons == []
    assert calls["count"] == 2


def test_d_state_processes_ignore_kernel_threads(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    kernel = proc / "2"
    user = proc / "42"
    kernel.mkdir(parents=True)
    user.mkdir()
    (kernel / "status").write_text("Name:\tjbd2/test\nState:\tD (disk sleep)\nPPid:\t2\n")
    (user / "status").write_text("Name:\tpython\nState:\tD (disk sleep)\nPPid:\t1\n")
    original_iterdir = Path.iterdir

    def fake_iterdir(self):
        if str(self) == "/proc":
            return original_iterdir(proc)
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    assert _d_state_processes() == [{"pid": "42", "name": "python", "ppid": "1", "state": "D (disk sleep)"}]


def test_peft_module_traversal_failure_is_retryable(tmp_path):
    stdout = tmp_path / "run.stdout.log"
    stderr = tmp_path / "run.stderr.log"
    stdout.write_text("Using model backend=dinov3_feature\n")
    stderr.write_text(
        '  File "/env/lib/python3.12/site-packages/torch/nn/modules/module.py", '
        "line 2797, in named_children\n"
        "ValueError: not enough values to unpack (expected 2, got 1)\n"
    )

    assert _is_transient_failure(1, False, stdout, stderr)
