"""Regression tests for the single active v9d arrow runtime policy."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from vla_benchmarking import run_arrow_pick_place_eval as eval_runner
from vla_benchmarking import run_arrow_pick_place_matrix as matrix_runner
from vla_benchmarking.controller_configs import (
    ACTIVE_CONTROLLER_NAME,
    ControllerConfigError,
    load_controller_config,
)


EXPECTED_V9D_HASH = "60f4f5f9ecfde7b4830f376ab06cfc706e2ef175d86817c42a0adb7cddd46c0c"


def test_default_load_is_flat_v9d_and_hash_is_preserved():
    config = load_controller_config()
    assert config["name"] == ACTIVE_CONTROLLER_NAME
    assert "extends" not in config
    assert config["config_hash"] == EXPECTED_V9D_HASH
    assert config["grasp_search"]["strategy"] == "rgbd_region"


def test_standard_runners_do_not_import_archived_zerograsp():
    probe = (
        "import sys; "
        "import vla_benchmarking.run_arrow_pick_place_eval; "
        "import vla_benchmarking.run_arrow_pick_place_matrix; "
        "assert 'vla_benchmarking.zerograsp_contracts' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_old_config_is_preserved_but_rejected_before_runtime():
    retired = (
        Path(__file__).parents[1]
        / "controller_configs"
        / "retired"
        / "v9g_post_lift_retention.json"
    )
    assert retired.is_file()
    with pytest.raises(ControllerConfigError, match="retired|cannot be executed"):
        load_controller_config(retired)


def test_eval_and_matrix_defaults_materialize_v9d_provenance(tmp_path: Path):
    resolved = eval_runner._resolve_controller_variant(None, suite_mode="vanilla")
    assert resolved.name == ACTIVE_CONTROLLER_NAME
    assert resolved.grasp_search is not None and resolved.grasp_search.enabled
    assert eval_runner._resolve_controller_variant("default", suite_mode="vanilla").canonical() == resolved.canonical()

    cell = matrix_runner.plan_cells(
        task_ids=[0], episodes_per_task=1, output_root=tmp_path
    )[0]
    assert cell["controller_variant"] == ACTIVE_CONTROLLER_NAME
    assert cell["controller_config_hash"] == EXPECTED_V9D_HASH
    assert cell["controller_config_path"].endswith(
        "controller_configs/v9d_rgbd_region_grasp_search.json"
    )


def test_retired_label_is_rejected_before_environment_construction():
    with pytest.raises(ControllerConfigError, match="only the active v9d"):
        eval_runner.build_libero_env(
            0,
            1000,
            256,
            controller_variant="libero_spatial_akita_bowl_agentview_v9g_post_lift_retention",
        )
    with pytest.raises(ValueError, match="only the active v9d"):
        matrix_runner.plan_cells(
            task_ids=[0], episodes_per_task=1, controller_variant="rgbd_arrow_v2"
        )


def test_zerograsp_provider_is_rejected_before_variant_construction():
    with pytest.raises(ControllerConfigError, match="ZeroGrasp.*retired"):
        eval_runner.controller_variant_from_config(
            {
                "name": ACTIVE_CONTROLLER_NAME,
                "grasp_provider": "zerograsp",
                "zerograsp": {"checkpoint": "side-experiment"},
            }
        )


@pytest.mark.parametrize(
    "retired_name",
    (
        "libero_spatial_akita_bowl_agentview_v9g_post_lift_retention",
        "libero_spatial_akita_bowl_agentview_v10_zg_grasp_only",
    ),
)
def test_runtime_resolver_rejects_retired_names(retired_name: str):
    with pytest.raises(ControllerConfigError, match="only the active v9d"):
        eval_runner._resolve_controller_variant(retired_name, suite_mode="vanilla")


def test_runtime_resolver_rejects_same_name_modified_policy():
    modified = eval_runner.ControllerVariantConfig(name=ACTIVE_CONTROLLER_NAME)
    with pytest.raises(ControllerConfigError, match="payload does not match"):
        eval_runner._resolve_controller_variant(modified, suite_mode="vanilla")
