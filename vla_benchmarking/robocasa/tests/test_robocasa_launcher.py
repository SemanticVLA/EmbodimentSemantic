from pathlib import Path


LAUNCHER = (
    Path(__file__).parents[1]
    / "arrow_grasp_controller"
    / "legion"
    / "run_robocasa_pickplace.sbatch"
)
REQUIREMENTS = Path(__file__).parents[1] / "requirements.txt"


def test_launcher_reuses_runtime_by_default_and_bootstraps_only_when_requested():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'SETUP_RUNTIME="${ROBOCASA_SETUP_RUNTIME:-0}"' in text
    assert 'MODE="${ROBOCASA_MODE:-}"' in text
    assert 'MODE=full' in text and 'MODE=smoke' in text
    assert 'ROBOCASA_MODE=full requires ROBOCASA_TASKS=all' in text
    assert 'ROBOCASA_MODE=full requires ROBOCASA_EPISODES_PER_TASK=1' in text
    assert 'ROBOCASA_MODE=full requires ROBOCASA_SEED_BASE=1000' in text
    assert '--mode "$MODE"' in text
    assert '[[ "$SETUP_RUNTIME" == 1 ]]' in text
    assert 'ROBOCASA_SETUP_RUNTIME=1 for bootstrap' in text
    assert '"$PYTHON" -m pip install --requirement "$REQUIREMENTS_FILE"' in text
    assert 'export TRANSFORMERS_CACHE="${ROBOCASA_TRANSFORMERS_CACHE:-$HF_HOME/transformers}"' in text
    assert 'GRASP_PROFILE="${ROBOCASA_GRASP_PROFILE:-canonical_rim}"' in text
    assert 'object_contact_v1) GRASP_CONFIG="$CONTROLLER_ROOT/configs/object_contact_v1.json"' in text
    assert '--grasp-profile "$GRASP_PROFILE"' in text

    requirements = REQUIREMENTS.read_text(encoding="utf-8")
    assert "torch==2.7.1" in requirements
    assert "torchvision==0.22.1" in requirements


def test_launcher_checks_pins_source_revisions_assets_and_motion_probe_before_eval():
    text = LAUNCHER.read_text(encoding="utf-8")

    for expected in (
        '"mujoco": "3.3.1"',
        '"lerobot": "0.3.3"',
        '"torch": "2.7.1"',
        '"torchvision": "0.22.1"',
        '"robosuite": "1.5.2"',
        '"robocasa": "1.0.1"',
        '"5ce6643f3092639d08f7b0f90ed1c6a84f50552c"',
        '"4f8a2980def75a55dff96b990745b83540425f09"',
        'direct_url.json',
        'assets_marker_matches',
        'ASSET_MARKER="$RUNTIME_ROOT/assets_ready_robocasa_${ROBOCASA_REVISION}"',
        'robocasa_revision=$ROBOCASA_REVISION',
        'grasp_profile=$GRASP_PROFILE',
        'grasp_config_sha256=$GRASP_CONFIG_SHA256',
        'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
        'runtime_probe.json',
        '--execute-motion',
    ):
        assert expected in text

    assert text.index("verify_assets") < text.index("probe_runtime") < text.rindex('"$ENTRYPOINT_MODULE"')
