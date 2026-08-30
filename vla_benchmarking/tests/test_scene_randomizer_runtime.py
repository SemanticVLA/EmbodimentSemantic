import numpy as np
import pytest
import json

from radomize_scenes import SceneRandomizerVecEnvWrapper, get_object_pose
from run_lora_2x2_eval import build_manifest, validate_randomization_audit


class _Model:
    nbody = 5
    njnt = 5
    nq = 35
    nv = 30
    jnt_type = [0] * 5
    jnt_qposadr = [0, 7, 14, 21, 28]
    jnt_dofadr = [0, 6, 12, 18, 24]
    def __init__(self):
        self._names = [
            "akita_black_bowl_1_main",
            "plate_1_main",
            "akita_black_bowl_2_main",
            "cookies_1_main",
            "glazed_rim_porcelain_ramekin_1_main",
        ]

    def body_id2name(self, body_id):
        return self._names[body_id]

    def body_name2id(self, name):
        return self._names.index(name)

    @property
    def body_jntadr(self):
        return list(range(5))


class _Data:
    qpos = np.zeros(35, dtype=float)
    qvel = np.zeros(30, dtype=float)


class _Sim:
    def __init__(self):
        self.model = _Model()
        self.data = _Data()

    def forward(self):
        return None

    def get_state(self):
        return np.concatenate((self.data.qpos, self.data.qvel))


class _Inner:
    def __init__(self):
        self.sim = _Sim()
        self.env = self

    def _get_observations(self, force_update=True):
        return np.zeros(1)


class _LeRobotEnv:
    def __init__(self):
        self._env = _Inner()
        self._init_state_id = 0
        self._init_states = np.zeros((1, 4), dtype=np.float64)

    def _format_raw_obs(self, raw):
        return raw


class _Vec:
    def __init__(self):
        self.envs = [_LeRobotEnv()]

    def reset(self, **kwargs):
        return np.zeros((1, 1))


class _Audit:
    def __init__(self):
        self.records = {}

    def log(self, **record):
        key = (record["task_id"], record.get("env_index", 0), record.get("reset_sequence", 0))
        current = self.records.setdefault(key, {
            "task_id": record["task_id"],
            "env_index": record.get("env_index", 0),
            "reset_sequence": record.get("reset_sequence", 0),
            "dimensions_enabled": {},
            "dimensions_realized": {},
            "details": {},
            "status": "pending",
        })
        current["dimensions_enabled"].update(record.get("dimensions_enabled", {}))
        current["dimensions_realized"].update(record.get("dimensions_realized", {}))
        current["details"].update(record.get("details", {}))
        if record.get("details", {}).get("status") == "environment_ok":
            current["status"] = "ok"


def _seed_poses():
    sim = _Sim()
    for index, label in enumerate(sim.model._names):
        sim.data.qpos[index * 7:index * 7 + 7] = [index, index + 0.1, index + 0.2, 1, 0, 0, 0]
    return sim


def test_reset_fails_when_configured_removed_object_is_present():
    vec = _Vec()
    vec.envs[0]._env.sim = _seed_poses()
    wrapper = SceneRandomizerVecEnvWrapper(
        vec, 3, {}, audit_logger=_Audit(), removal_config={3: ["glazed_rim_porcelain_ramekin_1"]}
    )
    with pytest.raises(RuntimeError, match="still present"):
        wrapper.reset()


def test_removal_only_reset_fails_when_protected_target_or_plate_is_missing():
    vec = _Vec()
    sim = _seed_poses()
    sim.model._names[0] = "other_target_main"
    vec.envs[0]._env.sim = sim
    wrapper = SceneRandomizerVecEnvWrapper(
        vec, 3, {}, audit_logger=_Audit(), removal_config={3: ["glazed_rim_porcelain_ramekin_1"]}
    )
    with pytest.raises(RuntimeError, match="protected object missing"):
        wrapper.reset()


def test_swap_fails_closed_when_required_object_is_missing():
    vec = _Vec()
    sim = _seed_poses()
    sim.model._names[3] = "other_cookies_main"
    sim.model._names[4] = "other_ramekin_main"
    vec.envs[0]._env.sim = sim
    wrapper = SceneRandomizerVecEnvWrapper(
        vec, 2, {2: [("akita_black_bowl_2", "cookies_1")]},
        audit_logger=_Audit(), removal_config={2: ["glazed_rim_porcelain_ramekin_1"]}
    )
    with pytest.raises(RuntimeError, match="layout was not fully applied"):
        wrapper.reset()


def test_swap_restores_protected_target_and_plate_poses():
    vec = _Vec()
    sim = _seed_poses()
    sim.model._names[4] = "other_ramekin_main"
    vec.envs[0]._env.sim = sim
    before = {
        label: get_object_pose(vec.envs[0]._env, label)
        for label in ("akita_black_bowl_1", "plate_1")
    }
    wrapper = SceneRandomizerVecEnvWrapper(
        vec, 2, {2: [("akita_black_bowl_2", "cookies_1")]},
        audit_logger=_Audit(), removal_config={2: ["glazed_rim_porcelain_ramekin_1"]}
    )
    wrapper.reset()
    for label, pose in before.items():
        np.testing.assert_allclose(get_object_pose(vec.envs[0]._env, label), pose)


def test_non_string_swap_operation_is_rejected():
    vec = _Vec()
    sim = _seed_poses()
    sim.model._names[4] = "other_ramekin_main"
    vec.envs[0]._env.sim = sim
    wrapper = SceneRandomizerVecEnvWrapper(
        vec,
        2,
        {2: [("akita_black_bowl_2", (0.0, -0.03, 0.0))]},
        audit_logger=_Audit(),
        removal_config={2: ["glazed_rim_porcelain_ramekin_1"]},
    )
    with pytest.raises(TypeError, match="string object-name pairs"):
        wrapper.reset()


def test_completed_swap_audit_record_validates_against_manifest(tmp_path):
    vec = _Vec()
    sim = _seed_poses()
    sim.model._names[4] = "other_ramekin_main"
    vec.envs[0]._env.sim = sim
    audit = _Audit()
    wrapper = SceneRandomizerVecEnvWrapper(
        vec, 2, {2: [("akita_black_bowl_2", "cookies_1")]},
        audit_logger=audit, removal_config={2: ["glazed_rim_porcelain_ramekin_1"]}
    )
    wrapper.reset()
    record = next(iter(audit.records.values()))
    manifest = build_manifest(
        base_checkpoint="base", treatment_checkpoint="treatment", seeds=[3], output_root=tmp_path
    )
    record["dimensions_enabled"] = manifest["randomization_dimensions"]["2"]
    # Prompt realization is sealed for every task; this runtime-only wrapper
    # record is completed with the manifest's prompt dimension before audit.
    record["dimensions_realized"]["prompt_variant"] = True
    record["details"]["status"] = "ok"
    records = [record]
    for task_id in manifest["tasks"]:
        if task_id == 2:
            continue
        dimensions = manifest["randomization_dimensions"][str(task_id)]
        details = {
            "removed": manifest["randomization_config"]["remove"][str(task_id)],
            "projection": {"success": True},
            "protected": {"akita_black_bowl_1": True, "plate_1": True},
        }
        if dimensions["scene_layout"]:
            layout = manifest["randomization_config"]["layout"][str(task_id)]
            details["layout"] = {
                "configured": layout,
                "applied": [label for operation in layout for label in operation],
                "skipped": [],
            }
        records.append({
            "task_id": task_id,
            "env_index": 0,
            "reset_sequence": 1,
            "dimensions_enabled": dimensions,
            "dimensions_realized": dimensions,
            "details": details,
            "status": "ok",
        })
    output = tmp_path / "cell"
    output.mkdir()
    (output / "randomization_audit.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )
    validate_randomization_audit(output, manifest)
