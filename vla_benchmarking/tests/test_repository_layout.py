from __future__ import annotations

import json
from pathlib import Path

from vla_benchmarking.arrow_grasp_controller.configs import load_controller_config


VLA_ROOT = Path(__file__).resolve().parents[1]


def test_vla_root_contains_no_loose_executable_source_files() -> None:
    loose = sorted(
        path.name
        for path in VLA_ROOT.iterdir()
        if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}
    )
    assert loose == []


def test_obsolete_compatibility_directories_are_absent() -> None:
    obsolete = {
        "arrow_policy",
        "calibrations",
        "controller_configs",
        "experiment_records",
        "grasp_controller",
        "legion",
        "molmo_sam3",
        "sanity_checks",
    }
    present = sorted(name for name in obsolete if (VLA_ROOT / name).exists())
    assert present == []


def test_policy_and_evaluation_packages_have_one_home() -> None:
    expected = [
        VLA_ROOT / "arrow_grasp_controller" / "controller",
        VLA_ROOT / "arrow_grasp_controller" / "configs",
        VLA_ROOT / "arrow_finetuned_vla" / "smolvla_no_arrows",
        VLA_ROOT / "arrow_finetuned_vla" / "smolvla_all_arrows",
        VLA_ROOT / "arrow_finetuned_vla" / "target_arrow_lora",
        VLA_ROOT / "arrow_finetuned_vla" / "action_visual_lora",
        VLA_ROOT / "arrow_finetuned_vla" / "graph_text_lora",
        VLA_ROOT / "arrow_finetuned_vla" / "language_free_arrow_student",
        VLA_ROOT / "evaluation",
    ]
    missing = [str(path.relative_to(VLA_ROOT)) for path in expected if not path.is_dir()]
    assert missing == []


def test_canonical_87_percent_config_identity_remains_frozen() -> None:
    config_path = (
        VLA_ROOT
        / "arrow_grasp_controller"
        / "configs"
        / "canonical_molmo_rgbd_grasp.json"
    )
    lock_path = config_path.with_name("active_policy.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    loaded = load_controller_config(config_path)
    expected = "37497fd0b2f60346b9ffd1501ccc743046c7fa2370ef6fa9531a7204f69cc044"
    assert loaded["config_hash"] == expected
    assert lock["canonical_config_sha256"] == expected
