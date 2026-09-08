from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import torch
import pytest

from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows import run_lerobot_train


class _Meter:
    def __init__(self, value: float):
        self.value = value


class _Tracker:
    def __init__(self, loss: float, grad_norm: float):
        self.loss = loss
        self.grad_norm = grad_norm

        self.lr = 0.0

    def to_dict(self, use_avg: bool = True):
        assert use_avg is False
        return {"loss": self.loss, "grad_norm": self.grad_norm, "lr": self.lr}


class _TrainModule:
    def __init__(self, loss: float = 1.25, grad_norm: float = 0.5):
        self.loss = loss
        self.grad_norm = grad_norm
        parameter = torch.nn.Parameter(torch.ones(()))
        self.optimizer = torch.optim.AdamW([parameter], lr=5e-5, weight_decay=1e-5, betas=(0.9, 0.95), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: (step + 1) / 667 if step < 666 else 1.0
        )

    def update_policy(self, train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, lr_scheduler):
        lr_scheduler.step()
        tracker = _Tracker(self.loss, self.grad_norm)
        tracker.lr = optimizer.param_groups[0]["lr"]
        return tracker, {"ok": True}


def _set_expected(monkeypatch, updates=1):
    values = {
        "TRAIN_EXPECTED_UPDATES": str(updates), "TRAIN_EXPECTED_BASE_LR": "5e-5",
        "TRAIN_EXPECTED_WEIGHT_DECAY": "1e-5", "TRAIN_EXPECTED_BETAS": "[0.9,0.95]",
        "TRAIN_EXPECTED_EPS": "1e-8", "TRAIN_EXPECTED_GRAD_CLIP": "10.0",
        "TRAIN_EXPECTED_SCHEDULER": "cosine_decay_with_warmup", "TRAIN_EXPECTED_WARMUP": "666",
        "TRAIN_EXPECTED_DECAY_STEPS": "20000", "TRAIN_EXPECTED_DECAY_LR": "2.5e-6",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_runtime_evidence_observes_finite_training_metrics(tmp_path: Path, monkeypatch):
    callbacks = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(run_lerobot_train.torch.cuda, "is_available", lambda: False)
    _set_expected(monkeypatch)
    module = _TrainModule()
    output = tmp_path / "runtime.json"
    finalize = run_lerobot_train._install_runtime_evidence(module, output)
    tracker, payload = module.update_policy(None, None, None, module.optimizer, 10, None, module.scheduler)
    assert tracker.loss == 1.25 and payload == {"ok": True}
    finalize()
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["updates_observed"] == 1
    assert evidence["all_losses_finite"] is True
    assert evidence["all_grad_norms_finite"] is True
    assert evidence["last_loss"] == 1.25
    assert evidence["peak_cuda_allocated_bytes"] == 0
    assert evidence["attestation_status"] == "VERIFIED"
    assert evidence["optimizer_class"] == "torch.optim.adamw.AdamW"


def test_runtime_evidence_rejects_nonfinite_loss(tmp_path: Path, monkeypatch):
    callbacks = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(run_lerobot_train.torch.cuda, "is_available", lambda: False)
    _set_expected(monkeypatch)
    module = _TrainModule(loss=float("nan"))
    run_lerobot_train._install_runtime_evidence(module, tmp_path / "runtime.json")
    try:
        module.update_policy(None, None, None, module.optimizer, 10, None, module.scheduler)
    except RuntimeError as exc:
        assert "non-finite training metric" in str(exc)
    else:
        raise AssertionError("non-finite loss was accepted")


def test_runtime_evidence_requires_full_expected_contract(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_lerobot_train.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="TRAIN_EXPECTED_UPDATES"):
        run_lerobot_train._install_runtime_evidence(_TrainModule(), tmp_path / "runtime.json")


def test_runtime_evidence_rejects_short_run_at_finalize(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_lerobot_train.torch.cuda, "is_available", lambda: False)
    _set_expected(monkeypatch, updates=2)
    module = _TrainModule()
    finalize = run_lerobot_train._install_runtime_evidence(module, tmp_path / "runtime.json")
    module.update_policy(None, None, None, module.optimizer, 10, None, module.scheduler)
    with pytest.raises(RuntimeError, match="runtime training evidence failed attestation"):
        finalize()
