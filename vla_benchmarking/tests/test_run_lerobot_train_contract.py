from __future__ import annotations

import atexit
import json
from pathlib import Path

import run_lerobot_train


class _Meter:
    def __init__(self, value: float):
        self.value = value


class _Tracker:
    def __init__(self, loss: float, grad_norm: float):
        self.loss = loss
        self.grad_norm = grad_norm

    def to_dict(self, use_avg: bool = True):
        assert use_avg is False
        return {"loss": self.loss, "grad_norm": self.grad_norm}


class _TrainModule:
    def __init__(self, loss: float = 1.25, grad_norm: float = 0.5):
        self.loss = loss
        self.grad_norm = grad_norm

    def update_policy(self, *args, **kwargs):
        return _Tracker(self.loss, self.grad_norm), {"ok": True}


def test_runtime_evidence_observes_finite_training_metrics(tmp_path: Path, monkeypatch):
    callbacks = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(run_lerobot_train.torch.cuda, "is_available", lambda: False)
    module = _TrainModule()
    output = tmp_path / "runtime.json"
    run_lerobot_train._install_runtime_evidence(module, output)
    tracker, payload = module.update_policy()
    assert tracker.loss == 1.25 and payload == {"ok": True}
    callbacks[0]()
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["updates_observed"] == 1
    assert evidence["all_losses_finite"] is True
    assert evidence["all_grad_norms_finite"] is True
    assert evidence["last_loss"] == 1.25
    assert evidence["peak_cuda_allocated_bytes"] == 0


def test_runtime_evidence_rejects_nonfinite_loss(tmp_path: Path, monkeypatch):
    callbacks = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(run_lerobot_train.torch.cuda, "is_available", lambda: False)
    module = _TrainModule(loss=float("nan"))
    run_lerobot_train._install_runtime_evidence(module, tmp_path / "runtime.json")
    try:
        module.update_policy()
    except RuntimeError as exc:
        assert "non-finite training metric" in str(exc)
    else:
        raise AssertionError("non-finite loss was accepted")
