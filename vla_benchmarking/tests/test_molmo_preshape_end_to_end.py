from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from molmo_sam3 import preshape
from run_arrow_pick_place_eval import _ActionBudget
from run_molmo_sam3_canary import (
    AGENTVIEW,
    GraspCandidate,
    run_canary_episode,
    build_scientific_identity,
    resolve_opening_profile,
)


_BASH = r"C:\Program Files\Git\bin\bash.exe" if os.path.isfile(r"C:\Program Files\Git\bin\bash.exe") else shutil.which("bash")


class _RobotEnv:
    """Small robot-only environment exercising the real preshape dispatch path."""

    def __init__(self, budget: int = 1200):
        self.observation = {
            "eef_pos": np.asarray((0.10, 0.20, 0.40), dtype=float),
            "eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0), dtype=float),
        }
        self._molmo_sam3_action_budget = _ActionBudget(budget)
        self.actions: list[np.ndarray] = []

    def _get_observations(self, force_update: bool = True):
        del force_update
        return self.observation

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float).copy())
        return self.observation, 0.0, False, {}


def test_real_preshape_dispatch_handles_delayed_settling_and_shared_budget(tmp_path):
    env = _RobotEnv()
    # Initial/close/holds: the first five holds are deliberately coarse and
    # leave the jaw above the target; a second five-hold settling round must
    # stabilize before completion.  No action-generation or step mock is used.
    readings = iter((0.050, 0.048, 0.0460, 0.0455, 0.0452, 0.0440, 0.0439,
                     0.0400, 0.0401, 0.0400, 0.0400, 0.0400))
    callback_calls: list[bool] = []
    audit = preshape.perform_preshape(
        env,
        measure_opening_fn=lambda _env: next(readings),
        output_dir=tmp_path,
        motion_started_callback=lambda: callback_calls.append(True),
    )

    assert audit["status"] == "completed"
    assert audit["actions_sent"] == 11  # one close + two five-hold rounds
    assert audit["budget_used"] == 11 == env._molmo_sam3_action_budget.used
    assert callback_calls == [True]
    assert [float(action[-1]) for action in env.actions] == [1.0] + [0.0] * 10
    assert all(0.035 <= value <= 0.045 for value in audit["settled_readings_m"])
    assert max(audit["settled_readings_m"]) - min(audit["settled_readings_m"]) <= 0.00025
    persisted = json.loads((tmp_path / "preshape_audit.json").read_text(encoding="utf-8"))
    assert persisted["actions_sent"] == 11


def test_below_band_after_real_hold_fails_closed_without_duplicate_action(tmp_path):
    env = _RobotEnv()
    readings = iter((0.050, 0.040, 0.034))
    with pytest.raises(preshape.PreshapeError) as error:
        preshape.perform_preshape(
            env, measure_opening_fn=lambda _env: next(readings), output_dir=tmp_path,
        )

    audit = error.value.audit
    assert error.value.category == "below_band"
    assert audit["status"] == "failed"
    assert audit["actions_sent"] == 2
    assert audit["budget_used"] == 2 == env._molmo_sam3_action_budget.used
    assert len(audit["actions"]) == 2
    assert [item["command"] for item in audit["actions"]] == ["close", "hold"]
    assert env._molmo_sam3_action_count == 2


def test_budget_exhaustion_charges_failed_hold_and_preserves_audit(tmp_path):
    env = _RobotEnv(budget=1)
    readings = iter((0.050, 0.048))
    with pytest.raises(preshape.PreshapeError) as error:
        preshape.perform_preshape(
            env, measure_opening_fn=lambda _env: next(readings), output_dir=tmp_path,
        )

    assert error.value.category == "timeout"
    audit = error.value.audit
    assert audit["actions_sent"] == 1
    assert audit["budget_used"] == 1
    assert env._molmo_sam3_action_count == 1
    assert len(audit["actions"]) == 1
    assert audit["actions"][0]["command"] == "close"


def test_helper_hard_cap_is_160_even_with_remaining_global_budget(tmp_path):
    env = _RobotEnv(budget=1200)
    with pytest.raises(preshape.PreshapeError) as error:
        preshape.perform_preshape(
            env, measure_opening_fn=lambda _env: 0.050, output_dir=tmp_path,
        )

    assert error.value.category == "timeout"
    assert error.value.audit["actions_sent"] == preshape.MAX_ACTIONS == 160
    assert error.value.audit["budget_used"] == 160
    assert env._molmo_sam3_action_budget.used == 160
    assert env._molmo_sam3_action_count == 160


def test_opening_profile_is_agentview_rgbd_only_and_identity_strictly_differs():
    full = resolve_opening_profile("full_open", region_backend="rgbd", camera_name=AGENTVIEW)
    shaped = resolve_opening_profile("preshape40mm", region_backend="rgbd", camera_name=AGENTVIEW)
    assert full["target_opening_m"] is None
    assert shaped["target_opening_m"] == pytest.approx(0.040)
    assert shaped["accepted_opening_band_m"] == (0.035, 0.045)
    with pytest.raises(ValueError):
        resolve_opening_profile("preshape40mm", region_backend="sam3", camera_name=AGENTVIEW)
    with pytest.raises(ValueError):
        resolve_opening_profile("preshape40mm", region_backend="rgbd", camera_name="wrist")

    common = dict(
        execution_sha="sha", controller_config_digest="cfg", model_provenance={},
        variant="molmo_dense_agentview", candidate_policy="dense", camera_name=AGENTVIEW,
        region_backend="rgbd", backend="v9d_rgbd_region", motion_profile="baseline",
        motion_profile_params={}, motion_diagnostics=False, observation_profile="baseline",
        observation_profile_params={}, task_ids=(4, 6), seed_base=1000,
        suite_modes=("vanilla", "sealed_randomized"),
    )
    _, full_hash = build_scientific_identity(**common, opening_profile="full_open", opening_profile_params=full)
    payload, shaped_hash = build_scientific_identity(
        **common, opening_profile="preshape40mm", opening_profile_params=shaped,
    )
    assert full_hash != shaped_hash
    assert payload["opening"]["profile"] == "preshape40mm"
    assert payload["opening"]["params"]["max_actions"] == 160


def test_retry_shapes_before_fresh_capture_and_proposal(tmp_path):
    """Exercise the real canary retry seam, including the preshape callback order."""
    events: list[str] = []

    def capture(label: str):
        calibration = SimpleNamespace(camera_name=AGENTVIEW)
        return SimpleNamespace(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            metric_depth=np.ones((4, 4), dtype=np.float32),
            calibration=calibration,
            label=label,
        )

    captures = [capture("initial")]

    class Worker:
        def propose(self, request):
            events.append(f"propose:{request.agentview_capture.label}")
            return [GraspCandidate(
                candidate_id=f"candidate-{len(events)}",
                position_world_m=(0.1, 0.2, 0.3),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                opening_m=0.04,
            )]

    def capture_fn(_env, *, resolution, camera_name):
        assert resolution == 4 and camera_name == AGENTVIEW
        events.append("capture")
        return captures[-1]

    def before_capture(_env, attempt_index):
        events.append(f"preshape:{attempt_index}")
        fresh = capture(f"retry-{attempt_index}")
        captures.append(fresh)
        return fresh

    calls = []

    def runner(*, context, evaluator, **kwargs):
        del evaluator, kwargs
        calls.append(context.attempt_index)
        if context.attempt_index == 1:
            return {"status": "grasp_failed", "grasp_retained": False, "retreat_complete": True}
        context.mark_retreat_complete()
        return {"status": "placed", "grasp_retained": True, "retreat_complete": True}

    result = run_canary_episode(
        env=object(), task_id=4, seed=1000, output_dir=tmp_path,
        variant="molmo_dense_agentview", worker=Worker(), episode_runner=runner,
        source_uv=(1, 1), capture_fn=capture_fn, resolution=4, dry_run=False,
        before_capture_fn=before_capture,
        before_propose_callback=lambda _env: events.append("probe"),
    )

    assert calls == [1, 2]
    assert result["final_result"]["status"] == "placed"
    assert events == ["capture", "probe", "propose:initial", "preshape:2", "probe", "propose:retry-2"]


@pytest.mark.skipif(_BASH is None, reason="bash is required for sbatch context rendering")
@pytest.mark.parametrize(
    ("opening_probe", "expected_profile"),
    ((None, "baseline"), ("0", "baseline"), ("1", "release_plus20mm")),
)
def test_sbatch_context_rendering_does_not_expand_shell_conditionals(tmp_path, opening_probe, expected_profile):
    """Render the actual job-context heredoc under set -u for all modes."""
    sbatch = (
        Path(__file__).parents[1] / "legion" / "v9d_molmo_campaign.sbatch"
    ).read_text(encoding="utf-8")
    start = sbatch.index('opening_probe="${V9D_MOLMO_OPENING_PROBE:-0}"')
    end = sbatch.index("\nEOF", start) + len("\nEOF")
    heredoc_block = sbatch[start:end]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_NAME": "v9d_molmo_campaign",
        "CANARY_LABEL": "qa",
        "CANARY_EXPECTED_COMMIT": "0" * 40,
        "REPO_ROOT": "/tmp/repo",
        "CAMPAIGN_SCRIPT": "/tmp/repo/campaign.py",
        "campaign_sha256": "c" * 64,
        "REQUIREMENTS_FILE": "/tmp/repo/requirements.txt",
        "requirements_sha256": "r" * 64,
        "actual_commit": "a" * 40,
        "BASE_PYTHON": "/tmp/python",
        "PYTHON": "/tmp/python",
        "RUNTIME_ROOT": "/tmp/runtime",
            # Resolve the heredoc in the pytest scratch cwd, matching the
            # launcher’s absolute RUN_ROOT contract without touching checkout.
            "RUN_ROOT": ".",
        "OUTPUT_ROOT": "/tmp/output",
        "ARCHIVE_ROOT": "/tmp/archive",
        "HF_HOME": "/tmp/hf",
        "setup_started_epoch": "10",
        "setup_finished_epoch": "12",
    }
    for key in ("V9D_MOLMO_OPENING_PROBE", "V9D_MOLMO_OBSERVATION_PROFILE", "V9D_MOLMO_MOTION_PROFILE"):
        env.pop(key, None)
    if opening_probe is not None:
        env["V9D_MOLMO_OPENING_PROBE"] = opening_probe
    if opening_probe == "1":
        env["V9D_MOLMO_OBSERVATION_PROFILE"] = "baseline"
        env["V9D_MOLMO_MOTION_PROFILE"] = "baseline"
    command = (
        "set -Eeuo pipefail; "
        + heredoc_block
        + "\ncat \"$RUN_ROOT/job_context.env\""
    )
    completed = subprocess.run([_BASH, "-uc", command], cwd=tmp_path, env=env, text=True, capture_output=True)
    context_file = tmp_path / "job_context.env"
    if context_file.is_file():
        context_file.unlink()
    assert completed.returncode == 0, completed.stderr
    rendered = completed.stdout
    assert "if [[" not in rendered
    assert "opening_probe=" + (opening_probe or "0") in rendered
    assert "observation_profile=" + expected_profile.replace("release_plus20mm", "hover20mm") in rendered
    assert "motion_profile=" + expected_profile in rendered
