from __future__ import annotations

import json
import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from vla_benchmarking.arrow_finetuned_vla.workflows import adapter_audit


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "arrow_finetuned_vla" / "workflows"
GRAPH_POLICY_ROOT = ROOT / "arrow_finetuned_vla" / "graph_text_lora"
EVALUATION_ROOT = ROOT / "evaluation"


def _bash_path(path: Path) -> str:
    """Translate a Windows path to the WSL path understood by bash.exe."""
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        return f"/mnt/{value[0].lower()}{value[2:]}"
    return value


def _executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | 0o111)


def _seed_resume_checkpoint(pretrained: Path) -> None:
    """Materialize the minimum real checkpoint-root shape for shell tests."""
    pretrained.mkdir(parents=True, exist_ok=True)
    (pretrained / "adapter_model.safetensors").write_bytes(b"adapter")
    (pretrained / "adapter_config.json").write_text("{}", encoding="utf-8")
    (pretrained / "adapter_audit.json").write_text(
        json.dumps({"expected_inventory_sha256": "fake-inventory"}), encoding="utf-8"
    )
    state = pretrained.parent / "training_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "optimizer.pt").write_bytes(b"optimizer")
    (state / "scheduler.pt").write_bytes(b"scheduler")
    (state / "random_states_0.pkl").write_bytes(b"rng")


def _expected_inventory_fixture(base_policy: str) -> dict:
    modules = sorted(adapter_audit.REQUIRED_ACTION_MODULES | {
        "model.vlm_with_expert.lm_expert.layers.0.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.v_proj",
    })
    names = sorted(
        [f"{module}.lora_A.default.weight" for module in modules]
        + [f"{module}.lora_B.default.weight" for module in modules]
    )
    value = {
        "schema_version": 1,
        "base_policy": base_policy,
        "base_policy_revision": adapter_audit.PINNED_BASE_POLICY_REVISION,
        "target_regex": adapter_audit.ACTION_SIDE_TARGET_REGEX,
        "peft_type": "LORA",
        "rank": 16,
        "modules_to_save": [],
        "effective_peft_config": {
            "peft_type": "LORA",
            "r": 16,
            "target_modules": adapter_audit.ACTION_SIDE_TARGET_REGEX,
            "modules_to_save": [],
        },
        "matched_module_names": modules,
        "trainable_parameter_names": names,
        "trainable_parameter_count": len(names),
        "total_parameter_count": len(names) + 1,
        "trainable_parameter_numel": len(names),
        "total_parameter_numel": len(names) + 1,
        "no_vision_backbone_trainables": True,
    }
    value["inventory_sha256"] = adapter_audit.inventory_sha256(value)
    return value


def _runtime(tmp_path: Path, *, real_launcher: bool = True) -> tuple[Path, dict[str, str], Path]:
    """Create a no-network operator sandbox with a traceable launcher boundary."""
    shutil.copy2(WORKFLOW_ROOT / "run_smolvla_pipeline.sh", tmp_path / "run_smolvla_pipeline.sh")
    if real_launcher:
        shutil.copy2(WORKFLOW_ROOT / "launch_lora_treatment.sh", tmp_path / "launch_lora_treatment.sh")
    else:
        _executable(
            tmp_path / "launch_lora_treatment.sh",
            f"#!/usr/bin/env bash\nset -eu\nprintf 'launcher\\n' >> \"{_bash_path(tmp_path / 'trace.log')}\"\n",
        )

    # Resume preflight invokes the adapter-audit boundary before training. A
    # deterministic test double keeps these operator tests dependency-free
    # while exercising the real launch ordering and arguments.
    (tmp_path / "adapter_audit.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "def load_expected_inventory(path):\n"
        "    value = json.loads(Path(path).read_text(encoding='utf-8'))\n"
        "    if not value.get('inventory_sha256') or value.get('base_policy_revision') != '6721902bc4d61e50a3bfdb11dfb4cb626f05d102': raise ValueError('invalid fixture inventory')\n"
        "    return value\n"
        "if '--checkpoint' in sys.argv: raise SystemExit(0)\n",
        encoding="utf-8",
    )

    trace = tmp_path / "trace.log"
    fixture_base = _bash_path(tmp_path / "base-policy")
    expected_fixture = _expected_inventory_fixture(fixture_base)
    (tmp_path / "expected_inventory_fixture.json").write_text(
        json.dumps(expected_fixture, sort_keys=True) + "\n", encoding="utf-8"
    )
    _executable(
        tmp_path / "lambda_preflight.sh",
        f"#!/usr/bin/env bash\nset -eu\nprintf 'preflight:%s:%s:%s\\n' \"$1\" \"${{STEPS_VALUE-}}\" \"${{SAVE_FREQ_VALUE-}}\" >> \"{_bash_path(trace)}\"\n",
    )
    _executable(
        tmp_path / "train_lora.sh",
        f"#!/usr/bin/env bash\nset -eu\nprintf 'train:%s:%s:%s:%s\\n' \"${{STEPS-}}\" \"${{SAVE_FREQ-}}\" \"${{SEED-}}\" \"${{PEFT_R-}}\" >> \"{_bash_path(trace)}\"\nif [[ -f \"$(dirname \"$0\")/resume_fail.flag\" && \"${{TRAINING_MODE-}}\" == resume ]]; then exit 42; fi\ncheckpoint_id=$(printf '%06d' \"$STEPS\")\nmkdir -p \"$OUTPUT_DIR/checkpoints/$checkpoint_id/pretrained_model\" \"$OUTPUT_DIR/checkpoints/$checkpoint_id/training_state\"\nprintf 'stub adapter' > \"$OUTPUT_DIR/checkpoints/$checkpoint_id/pretrained_model/adapter_model.safetensors\"\nprintf '{{}}' > \"$OUTPUT_DIR/checkpoints/$checkpoint_id/pretrained_model/adapter_config.json\"\nprintf '{{\"expected_inventory_sha256\":\"{expected_fixture['inventory_sha256']}\"}}' > \"$OUTPUT_DIR/checkpoints/$checkpoint_id/pretrained_model/adapter_audit.json\"\nprintf 'optimizer' > \"$OUTPUT_DIR/checkpoints/$checkpoint_id/training_state/optimizer.pt\"\nprintf 'scheduler' > \"$OUTPUT_DIR/checkpoints/$checkpoint_id/training_state/scheduler.pt\"\nprintf 'rng' > \"$OUTPUT_DIR/checkpoints/$checkpoint_id/training_state/random_states_0.pkl\"\ncp \"$(dirname \"$0\")/expected_inventory_fixture.json\" \"$OUTPUT_DIR/expected_adapter_inventory.json\"\n",
    )

    # Keep the default launcher paths populated as well: the operator passes a
    # sealed environment to its child launcher and must remain deterministic
    # even when the parent process has no training-specific variables.
    data_root = tmp_path / "data"
    data_root.mkdir()
    pair_manifest = data_root / "sealed_lora_pair_manifest.json"
    pair_manifest.write_text('{"pair_kind":"sealed_lora_control_treatment"}\n', encoding="utf-8")
    pair_sentinel = data_root / "sealed_lora_pair_verified.json"
    pair_sentinel.write_text('{"full_experiment_ready":true}\n', encoding="utf-8")
    default_data_root = tmp_path / "lora_datasets"
    default_data_root.mkdir()
    shutil.copy2(pair_manifest, default_data_root / pair_manifest.name)
    shutil.copy2(pair_sentinel, default_data_root / pair_sentinel.name)
    fake_site = tmp_path / "fake_site"
    for package in ("peft", "safetensors"):
        (fake_site / package).mkdir(parents=True)
    (fake_site / "peft" / "__init__.py").write_text(
        "class PeftConfig:\n    @classmethod\n    def from_pretrained(cls, path):\n        return cls()\n",
        encoding="utf-8",
    )
    (fake_site / "safetensors" / "__init__.py").write_text(
        "class _Handle:\n    def __enter__(self): return self\n    def __exit__(self, *args): return False\n    def keys(self): return ['stub']\ndef safe_open(*args, **kwargs): return _Handle()\n",
        encoding="utf-8",
    )
    python_stub = tmp_path / "python_stub"
    _executable(
        python_stub,
        f"#!/usr/bin/env bash\nset -eu\nPYTHONPATH=\"{_bash_path(fake_site)}\" exec /usr/bin/python3 \"$@\"\n",
    )

    env = os.environ.copy()
    env.update(
        {
            "TRACE": _bash_path(trace),
            "DATA_ROOT": _bash_path(data_root),
            "PAIR_MANIFEST": _bash_path(pair_manifest),
            "PAIR_SENTINEL": _bash_path(pair_sentinel),
            "BASE_POLICY": _bash_path(tmp_path / "base-policy"),
            "PYTHONPATH": _bash_path(fake_site),
            # These values must not leak into the sealed schedule.
            "EPOCHS": "999",
            "STEPS": "777",
            "SAVE_FREQ": "13",
            "UPDATES_PER_EPOCH": "17",
            "SEED": "777",
            "PEFT_R": "3",
            # The copied operator script is intentionally run as an isolated
            # sandbox; point its organized roots at the adjacent test doubles.
            "REPO_ROOT": _bash_path(tmp_path),
            "VLA_ROOT": _bash_path(tmp_path),
        }
    )
    return tmp_path / "run_smolvla_pipeline.sh", env, trace


def _run(script: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    # Set organized roots inside the shell command so Windows->WSL environment
    # forwarding cannot silently replace the sandbox with repository defaults.
    # The Windows bash bridge also drops POSIX path-valued entries supplied
    # through subprocess(env=...).  Forward the fixture's path contract in the
    # command itself so the copied launcher cannot fall back to host defaults.
    assignments_by_key = {
        "REPO_ROOT": _bash_path(script.parent),
        "VLA_ROOT": _bash_path(script.parent),
    }
    for key in (
        "TRACE",
        "DATA_ROOT",
        "PAIR_MANIFEST",
        "PAIR_SENTINEL",
        "BASE_POLICY",
        "PYTHONPATH",
        "RUN_ROOT",
        "LIBERO_DATA_DIR",
        "LIBERO_DIR",
        "LIBERO_CONFIG",
        "LAMBDA_VENV",
    ):
        value = env.get(key)
        if value:
            assignments_by_key[key] = value
    assignments = tuple(
        f"{key}={shlex.quote(value)}" for key, value in assignments_by_key.items()
    )
    command = "env " + " ".join(assignments) + " bash " + shlex.quote(script.name)
    command += " " + " ".join(shlex.quote(arg) for arg in args)
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=script.parent,
        env=env,
        text=True,
        capture_output=True,
    )


def test_operator_pipeline_help_and_shell_syntax() -> None:
    script = WORKFLOW_ROOT / "run_smolvla_pipeline.sh"
    result = subprocess.run(
        ["bash", _bash_path(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "setup|dry|smoke|full|resume|eval" in result.stdout
    assert "--profile" in result.stdout
    for name in (
        "run_smolvla_pipeline.sh",
        "prepare_lambda_data.sh",
        "launch_lora_treatment.sh",
        "train_lora.sh",
        "lambda_preflight.sh",
        "bootstrap_lambda_runtime.sh",
        "launch_lora_pair.sh",
    ):
        checked = subprocess.run(["bash", "-n", _bash_path((WORKFLOW_ROOT / name) if name != "bootstrap_lambda_runtime.sh" else (ROOT / "environment" / name))], cwd=ROOT, text=True)
        assert checked.returncode == 0, name


def test_training_schedule_is_sealed_and_data_profiles_are_explicit() -> None:
    launch = (WORKFLOW_ROOT / "launch_lora_treatment.sh").read_text(encoding="utf-8")
    train = (WORKFLOW_ROOT / "train_lora.sh").read_text(encoding="utf-8")
    data = (WORKFLOW_ROOT / "prepare_lambda_data.sh").read_text(encoding="utf-8")
    preflight = (WORKFLOW_ROOT / "lambda_preflight.sh").read_text(encoding="utf-8")
    assert "SEALED_STEPS=$((SEALED_EPOCHS * SEALED_UPDATES_PER_EPOCH))" in launch
    assert "SEALED_UPDATES_PER_EPOCH=1946" in launch
    assert "ignore ambient EPOCHS/STEPS/SAVE_FREQ" in launch
    assert "TRAINING_MODE" in train
    assert "convert-target-arrow-pair" not in data
    assert "verify-target-arrow" not in data
    assert 'treatment) DATA_MODE="convert-pair"; VERIFY_MODE="verify"; PREFLIGHT_MODE="preflight"' in data
    assert 'no_arrow_treatment) DATA_MODE="convert-pair"; VERIFY_MODE="verify"; PREFLIGHT_MODE="preflight"' in data
    assert 'PREFLIGHT_MODE="preflight-target-arrow"' not in data
    assert "a0dded49581dcbf5a109f8350305411d345c5d99" in data
    assert "560fae66b5b41d2e383e6876b15052bbc105f4aae906cf1f630a2310b87a1fa9" in data
    assert 'REQUIRED_PAIR_VARIANT="target_arrow_treatment"' not in preflight
    assert 'git -C "$LIBERO_DIR" status --porcelain --untracked-files=no' in train
    assert 'LIBERO_COMMIT="${LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"' in train
    assert '"libero_worktree_status": "clean"' in train


def test_graph_profiles_are_explicitly_separated_and_use_the_sealed_graph_contract() -> None:
    launch = (WORKFLOW_ROOT / "launch_lora_treatment.sh").read_text(encoding="utf-8")
    train = (WORKFLOW_ROOT / "train_lora.sh").read_text(encoding="utf-8")
    preflight = (WORKFLOW_ROOT / "lambda_preflight.sh").read_text(encoding="utf-8")
    prepare = (WORKFLOW_ROOT / "prepare_lambda_data.sh").read_text(encoding="utf-8")

    for profile in ("graph_treatment", "arrow_graph_treatment"):
        assert profile in launch
        assert profile in train
        assert profile in preflight
        assert profile in prepare
    assert 'DATASET_VARIANT="graph_treatment"' in launch
    assert 'DATASET_VARIANT="arrow_graph_treatment"' in launch
    assert 'ADAPTER_KEY="graph_treatment_adapter"' in launch
    assert 'ADAPTER_KEY="arrow_graph_treatment_adapter"' in launch
    assert 'PAIR_KIND="sealed_lora_graph_treatment_arrow_graph_treatment"' in launch
    assert 'PREFLIGHT_VARIANT="graph_treatment"' in launch
    assert 'PREFLIGHT_VARIANT="arrow_graph_treatment"' in launch
    assert 'TRAINING_MODE' in launch
    assert 'TRAIN_ARGS+=(--policy.tokenizer_max_length=96)' in train
    assert 'CONVERTER_MODE="preflight-graph"' in preflight
    assert 'DATA_MODE="convert-graph-pair"' in prepare
    assert 'VERIFY_MODE="verify-graph"' in prepare


def test_graph_evaluation_routes_to_the_sealed_two_cell_evaluator() -> None:
    pipeline = (WORKFLOW_ROOT / "run_smolvla_pipeline.sh").read_text(encoding="utf-8")
    assert "run_lora_graph_pair_eval.py" in pipeline
    assert "arrow_graph_treatment is prepare-only and has no evaluation cell" in pipeline
    assert "PROFILE_CANONICAL" in pipeline
    assert 'PROFILE_CANONICAL" == graph_treatment' in pipeline


def test_graph_pilot_is_explicitly_non_confirmatory_and_preserves_paper_gate() -> None:
    evaluator = (GRAPH_POLICY_ROOT / "run_lora_graph_pair_eval.py").read_text(encoding="utf-8")
    launcher = (WORKFLOW_ROOT / "launch_lora_treatment.sh").read_text(encoding="utf-8")
    assert '"pilot": True, "confirmatory": False' in evaluator
    assert '"confirmatory_required_for_paper_claims": True' in evaluator
    assert '"status": "pilot_prepared_not_launchable"' in launcher
    assert '"confirmatory_required_for_paper_claims": True' in launcher


def test_lower_level_eval_rejects_graph_visual_arrow_bypasses() -> None:
    """The low-level wrapper must preserve the graph text-only contract."""
    evaluator = (EVALUATION_ROOT / "run_lerobot_eval_with_context.py").read_text(encoding="utf-8")
    assert "graph_treatment evaluation is sealed to no visual arrows" in evaluator
    assert "arrow_graph_treatment is prepare-only and cannot be evaluated" in evaluator
    assert "graph_profile" in evaluator


def test_historical_profiles_do_not_inherit_graph_tokenizer_or_adapter_metadata() -> None:
    train = (WORKFLOW_ROOT / "train_lora.sh").read_text(encoding="utf-8")
    launch = (WORKFLOW_ROOT / "launch_lora_treatment.sh").read_text(encoding="utf-8")
    assert 'if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then' in train
    assert 'if variant in ("graph_treatment", "arrow_graph_treatment"):' in launch
    # The 96-token override must be guarded by the graph-only branch, not a
    # global default that would silently alter treatment/no-arrow runs.
    graph_guard = 'if [[ "$VARIANT" == graph_treatment || "$VARIANT" == arrow_graph_treatment ]]; then'
    assert train.index(graph_guard) < train.index('TRAIN_ARGS+=(--policy.tokenizer_max_length=96)')
    assert 'tokenizer_max_length": 96' not in launch[: launch.index('if variant in ("graph_treatment", "arrow_graph_treatment"):')]


def test_graph_pair_verification_recomputes_the_current_formatter_digest() -> None:
    """A stored formatter hash is evidence only if verification re-hashes source."""
    converter = (WORKFLOW_ROOT / "hdf5_to_lerobot_dataset.py").read_text(encoding="utf-8")
    start = converter.index("def _load_sealed_manifest")
    end = converter.index("def validate_verified_pair", start)
    verifier = converter[start:end]
    assert "graph_formatter_sha256" in verifier
    assert "sha256_file(formatter_path)" in verifier


def test_resume_binds_the_complete_scheduled_checkpoint_root() -> None:
    launch = (WORKFLOW_ROOT / "launch_lora_treatment.sh").read_text(encoding="utf-8")
    train = (WORKFLOW_ROOT / "train_lora.sh").read_text(encoding="utf-8")
    for source in (launch, train):
        assert "pretrained_model" in source
        assert "training_state" in source
        assert "optimizer" in source and "scheduler" in source
        assert "random_state" in source or "rng" in source
        assert "1946" in source and "29190" in source
    assert "checkpoint_root_tree_sha256" in launch
    assert "checkpoint_root_inventory" in launch
    assert "expected_adapter_inventory.pending.json" in launch
    assert 'resume_checkpoint_root" == "$expected_checkpoint_root"' in launch
    assert 'resume_checkpoint_root" == "$resume_root/checkpoints/$resume_checkpoint_name"' in train


def test_active_training_and_eval_surfaces_have_no_target_arrow_profile() -> None:
    """The retired target-arrow profile must not remain operator-reachable."""
    active = (
        "run_smolvla_pipeline.sh",
        "bootstrap_lambda_runtime.sh",
        "prepare_lambda_data.sh",
        "prepare_base_snapshot.sh",
        "lambda_preflight.sh",
        "launch_lora_pair.sh",
        "launch_lora_treatment.sh",
        "train_lora.sh",
        "run_lerobot_train.py",
        "run_lora_2x2_eval.py",
    )
    forbidden = ("target-arrow", "target_arrow", "target arrow")
    for name in active:
        source_root = WORKFLOW_ROOT
        if name == "README.md":
            source_root = ROOT / "arrow_finetuned_vla"
        elif name == "bootstrap_lambda_runtime.sh":
            source_root = ROOT / "environment"
        text = (source_root / name).read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden), name


def test_parser_rejects_repeated_and_action_incompatible_options_without_invoking_stages(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path, real_launcher=False)

    repeated = _run(script, ["dry", "--profile", "treatment", "--profile", "no-arrow"], env)
    assert repeated.returncode != 0, repeated.stdout + repeated.stderr
    assert not trace.exists(), "repeated options must fail before any stage is invoked"

    incompatible = _run(script, ["dry", "--profile", "treatment", "--episodes", "2"], env)
    assert incompatible.returncode != 0, incompatible.stdout + incompatible.stderr
    assert not trace.exists(), "eval-only options must fail before any stage is invoked"


def test_setup_maps_no_arrow_profile_through_all_setup_stages(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path, real_launcher=False)
    (tmp_path / "environment").mkdir()
    for name in ("bootstrap_lambda_runtime.sh", "prepare_lambda_data.sh", "prepare_base_snapshot.sh"):
        target = tmp_path / "environment" / name if name == "bootstrap_lambda_runtime.sh" else tmp_path / name
        _executable(
            target,
            f"#!/usr/bin/env bash\nset -eu\nprintf '{name}:%s\\n' \"${{1-}}\" >> \"{_bash_path(trace)}\"\n",
        )
    # setup invokes its own preflight path after the data/base stages.
    _executable(
        tmp_path / "lambda_preflight.sh",
        f"#!/usr/bin/env bash\nset -eu\nprintf 'lambda_preflight.sh:%s\\n' \"$1\" >> \"{_bash_path(trace)}\"\n",
    )
    result = _run(script, ["setup", "--profile", "no-arrow"], env)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "bootstrap_lambda_runtime.sh:",
        "prepare_lambda_data.sh:no_arrow_treatment",
        "prepare_base_snapshot.sh:",
        "lambda_preflight.sh:no_arrow_treatment",
    ]


def test_dry_no_arrow_preflights_prints_and_writes_no_run_directory(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path)
    run_dir = tmp_path / "dry-run"
    result = _run(
        script,
        ["dry", "--profile", "no-arrow", "--run-dir", _bash_path(run_dir), "--python", _bash_path(tmp_path / "python_stub")],
        env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "preflight:no_arrow_treatment:29190:1946",
    ]
    assert "dry run complete" in result.stdout
    assert not run_dir.exists()
    assert not (tmp_path / "dry-run.training_plan.pending.json").exists()


def test_operator_target_arrow_profile_fails_before_any_stage_or_output(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path, real_launcher=False)
    run_dir = tmp_path / "must-not-be-created"
    result = _run(
        script,
        ["dry", "--profile", "target-arrow", "--run-dir", _bash_path(run_dir)],
        env,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert not trace.exists(), "retired profile must fail before invoking a stage"
    assert not run_dir.exists(), "retired profile must fail before deriving/writing output"


def test_direct_target_arrow_entry_points_fail_closed_before_side_effects(tmp_path: Path) -> None:
    """Reject the canonical internal profile name before network/runtime work."""
    trace = tmp_path / "unexpected-side-effect.log"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for command in ("curl", "wget", "python", "git", "sbatch"):
        _executable(
            fake_bin / command,
            f"#!/usr/bin/env bash\nprintf '{command}\\n' >> \"{_bash_path(trace)}\"\nexit 99\n",
        )
    # Pass the POSIX variables through an in-WSL `env` command.  Supplying
    # WSL paths only via subprocess(env=...) is not reliable with the Windows
    # bash.exe bridge: the child can silently see the host/default values.
    assignments = {
        "PATH": f"{_bash_path(fake_bin)}:/usr/bin:/bin",
        "PYTHON": "python",
        "DATA_ROOT": _bash_path(tmp_path / "data"),
        "LIBERO_DATA_DIR": _bash_path(tmp_path / "hdf5"),
        "LIBERO_DATASET_CACHE": _bash_path(tmp_path / "cache"),
        "LIBERO_DIR": _bash_path(tmp_path / "LIBERO"),
        "LIBERO_CONFIG": _bash_path(tmp_path / "libero-config.yaml"),
        "BASE_POLICY": _bash_path(tmp_path / "base-policy"),
        "RUN_ROOT": _bash_path(tmp_path / "run"),
        "OUTPUT_DIR": _bash_path(tmp_path / "output"),
        "HOME": _bash_path(tmp_path / "home"),
        "TRAINING_MODE": "smoke",
        "TRAINING_PROFILE": "target_arrow_treatment",
    }
    bash = shutil.which("bash")
    assert bash, "bash is required for shell-contract tests"
    commands = (
        ("prepare_lambda_data.sh", ("target_arrow_treatment",)),
        ("lambda_preflight.sh", ("target_arrow_treatment",)),
        ("launch_lora_treatment.sh", ("dry",)),
        ("train_lora.sh", ("target_arrow_treatment",)),
    )
    for script_name, args in commands:
        command = "env " + " ".join(
            f"{key}={shlex.quote(value)}" for key, value in assignments.items()
        )
        command += f" bash {shlex.quote(_bash_path(WORKFLOW_ROOT / script_name))}"
        command += " " + " ".join(shlex.quote(value) for value in args)
        result = subprocess.run(
            [bash, "-lc", command],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2, f"{script_name}: {result.stdout}{result.stderr}"
    assert not trace.exists(), "retired profile reached a network/runtime command"
    for path in ("data", "hdf5", "cache", "LIBERO", "home", "run", "output"):
        assert not (tmp_path / path).exists(), f"retired profile created {path}"


def test_smoke_is_exactly_two_steps_and_full_schedule_ignores_ambient_values(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path)
    smoke_dir = tmp_path / "smoke"
    smoke = _run(
        script,
        ["smoke", "--profile", "treatment", "--run-dir", _bash_path(smoke_dir), "--python", _bash_path(tmp_path / "python_stub")],
        env,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    smoke_lines = trace.read_text(encoding="utf-8").splitlines()
    assert "preflight:treatment:2:2" in smoke_lines
    assert any(line.startswith("train:2:2:1000:16") for line in smoke_lines)
    smoke_plan = json.loads((smoke_dir / "training_plan.json").read_text(encoding="utf-8"))
    assert smoke_plan["flags"]["steps"] == 2
    assert smoke_plan["flags"]["save_freq"] == 2

    trace.write_text("", encoding="utf-8")
    full_dir = tmp_path / "full"
    full = _run(
        script,
        ["full", "--profile", "treatment", "--run-dir", _bash_path(full_dir), "--python", _bash_path(tmp_path / "python_stub")],
        env,
    )
    assert full.returncode == 0, full.stdout + full.stderr
    full_lines = trace.read_text(encoding="utf-8").splitlines()
    assert "preflight:treatment:29190:1946" in full_lines
    assert any(line.startswith("train:29190:1946:1000:16") for line in full_lines)
    full_plan = json.loads((full_dir / "training_plan.json").read_text(encoding="utf-8"))
    assert full_plan["flags"]["steps"] == 29190
    assert full_plan["flags"]["save_freq"] == 1946
    assert full_plan["flags"]["epochs"] == 15
    assert full_plan["flags"]["updates_per_epoch"] == 1946


def test_resume_fail_closed_for_missing_outside_incompatible_and_completed_runs(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path)

    missing = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(tmp_path / "missing")], env)
    assert missing.returncode != 0
    assert not trace.exists() or "train:" not in trace.read_text(encoding="utf-8")

    run_dir = tmp_path / "interrupted"
    (run_dir / "checkpoints" / "001946").mkdir(parents=True)
    plan = {
        "training_variant": "treatment",
        "base_policy_revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        "flags": {"steps": 29190, "save_freq": 1946},
    }
    # Use the native test-side path for the digest; the shell receives the WSL
    # translation through env["PAIR_MANIFEST"].
    import hashlib

    pair_bytes = (tmp_path / "lora_datasets" / "sealed_lora_pair_manifest.json").read_bytes()
    plan["pair_manifest_sha256"] = hashlib.sha256(pair_bytes).hexdigest()
    revision = plan["base_policy_revision"]
    base_policy = tmp_path / "base_models" / f"smolvla_libero-{revision}"
    # The production launcher authenticates resume metadata against the
    # explicitly selected BASE_POLICY.  Keep the sandbox environment aligned
    # with the pending plan rather than relying on its unrelated default
    # fixture path.
    env["BASE_POLICY"] = _bash_path(base_policy)
    data_root = tmp_path / "lora_datasets"
    libero_dir = tmp_path / "LIBERO"
    env["LIBERO_DIR"] = _bash_path(libero_dir)
    libero_commit = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    plan.update(
        {
            "dataset_variant": "treatment",
            "dataset_repo_id": "local/libero_spatial_treatment",
            "base_policy": _bash_path(base_policy),
            "data_root": _bash_path(data_root),
            "libero_dir": _bash_path(libero_dir),
            "libero_commit": libero_commit,
            "libero_worktree_status": "clean",
            "libero_tracked_clean": True,
            "flags": {"seed": 1000, "peft_r": 16, "batch_size": 32, "steps": 29190, "save_freq": 1946},
        }
    )
    (run_dir / "training_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    outside = tmp_path / "outside-train-config.json"
    outside.write_text("{}", encoding="utf-8")
    outside_result = _run(
        script,
        ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir), "--resume-config", _bash_path(outside)],
        env,
    )
    assert outside_result.returncode != 0

    incompatible_plan = dict(plan)
    incompatible_plan["training_variant"] = "no_arrow_treatment"
    (run_dir / "training_plan.json").write_text(json.dumps(incompatible_plan), encoding="utf-8")
    inside = run_dir / "checkpoints" / "001946" / "pretrained_model" / "train_config.json"
    inside.parent.mkdir(parents=True, exist_ok=True)
    _seed_resume_checkpoint(inside.parent)
    inside.write_text("{}", encoding="utf-8")
    incompatible = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir)], env)
    assert incompatible.returncode != 0

    (run_dir / "training_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run_dir / "training_manifest.json").write_text("{}", encoding="utf-8")
    completed = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir)], env)
    assert completed.returncode != 0

    compatible = tmp_path / "compatible"
    (compatible / "checkpoints" / "001946" / "pretrained_model").mkdir(parents=True)
    _seed_resume_checkpoint(compatible / "checkpoints" / "001946" / "pretrained_model")
    plan["output_dir"] = _bash_path(compatible)
    (compatible / "checkpoints" / "001946" / "pretrained_model" / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "local/libero_spatial_treatment", "root": _bash_path(data_root / "treatment")},
                "output_dir": _bash_path(compatible),
                "seed": 1000,
                "batch_size": 32,
                "steps": 29190,
                "save_freq": 1946,
                "peft": {"r": 16},
            }
        ),
        encoding="utf-8",
    )
    pending = compatible.with_name(compatible.name + ".training_plan.pending.json")
    pending.write_text(json.dumps(plan), encoding="utf-8")
    (compatible.with_name(compatible.name + ".expected_adapter_inventory.pending.json")).write_text(
        json.dumps(_expected_inventory_fixture(_bash_path(base_policy))), encoding="utf-8"
    )
    pending_payload = pending.read_text(encoding="utf-8")
    (compatible.with_name(compatible.name + ".run_provenance.pending.json")).write_text(
        json.dumps(
            {
                "variant": "treatment",
                "dataset_variant": "treatment",
                "dataset_repo_id": "local/libero_spatial_treatment",
                    "base_policy": _bash_path(base_policy),
                    "base_policy_revision": revision,
                    "libero_dir": _bash_path(libero_dir),
                    "libero_commit": libero_commit,
                    "libero_worktree_status": "clean",
                    "libero_tracked_clean": True,
                    "pair_manifest_sha256": plan["pair_manifest_sha256"],
                "flags": {"seed": 1000, "peft_r": 16, "batch_size": 32, "steps": 29190, "save_freq": 1946},
            }
        ),
        encoding="utf-8",
    )
    interrupted = _run(
        script,
        ["resume", "--profile", "treatment", "--run-dir", _bash_path(compatible), "--python", _bash_path(tmp_path / "python_stub")],
        env,
    )
    assert interrupted.returncode == 0, interrupted.stdout + interrupted.stderr
    assert (compatible / "training_plan.json").is_file()
    assert (compatible / "training_plan.json").read_text(encoding="utf-8") == pending_payload
    assert not pending.exists()

    bad_provenance = tmp_path / "bad-preserved-provenance"
    (bad_provenance / "checkpoints" / "001946" / "pretrained_model").mkdir(parents=True)
    _seed_resume_checkpoint(bad_provenance / "checkpoints" / "001946" / "pretrained_model")
    bad_plan = dict(plan)
    bad_plan["output_dir"] = _bash_path(bad_provenance)
    (bad_provenance / "checkpoints" / "001946" / "pretrained_model" / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "local/libero_spatial_treatment", "root": _bash_path(data_root / "treatment")},
                "output_dir": _bash_path(bad_provenance),
                "seed": 1000,
                "batch_size": 32,
                "steps": 29190,
                "save_freq": 1946,
                "peft": {"r": 16},
            }
        ),
        encoding="utf-8",
    )
    (bad_provenance / "training_plan.json").write_text(json.dumps(bad_plan), encoding="utf-8")
    (bad_provenance / "run_provenance.json").write_text(
        json.dumps(
            {
                "variant": "treatment",
                "dataset_variant": "control",
                "dataset_repo_id": "local/libero_spatial_control",
                "base_policy": _bash_path(base_policy),
                "base_policy_revision": bad_plan["base_policy_revision"],
                "libero_dir": _bash_path(libero_dir),
                "libero_commit": libero_commit,
                "libero_worktree_status": "clean",
                "libero_tracked_clean": True,
                "pair_manifest_sha256": bad_plan["pair_manifest_sha256"],
                "flags": {"steps": 29190, "save_freq": 1946, "batch_size": 32, "seed": 1000, "peft_r": 16},
            }
        ),
        encoding="utf-8",
    )
    trace_before_bad = trace.read_text(encoding="utf-8")
    bad = _run(
        script,
        ["resume", "--profile", "treatment", "--run-dir", _bash_path(bad_provenance), "--python", _bash_path(tmp_path / "python_stub")],
        env,
    )
    assert bad.returncode != 0, bad.stdout + bad.stderr
    trace_after_bad = trace.read_text(encoding="utf-8")
    assert trace_after_bad.count("train:") == trace_before_bad.count("train:")

    mismatched_config = tmp_path / "mismatched-train-config"
    (mismatched_config / "checkpoints" / "001946" / "pretrained_model").mkdir(parents=True)
    _seed_resume_checkpoint(mismatched_config / "checkpoints" / "001946" / "pretrained_model")
    mismatched_plan = dict(plan)
    mismatched_plan["output_dir"] = _bash_path(mismatched_config)
    (mismatched_config / "training_plan.json").write_text(json.dumps(mismatched_plan), encoding="utf-8")
    (mismatched_config / "run_provenance.json").write_text(
        json.dumps(
            {
                "variant": "treatment",
                "dataset_variant": "treatment",
                "dataset_repo_id": "local/libero_spatial_treatment",
                "base_policy": _bash_path(base_policy),
                "base_policy_revision": revision,
                "libero_dir": _bash_path(libero_dir),
                "libero_commit": libero_commit,
                "libero_worktree_status": "clean",
                "libero_tracked_clean": True,
                "pair_manifest_sha256": plan["pair_manifest_sha256"],
                "flags": {"seed": 1000, "peft_r": 16, "batch_size": 32, "steps": 29190, "save_freq": 1946},
            }
        ),
        encoding="utf-8",
    )
    (mismatched_config / "checkpoints" / "001946" / "pretrained_model" / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "local/libero_spatial_treatment", "root": _bash_path(data_root / "treatment")},
                "output_dir": _bash_path(mismatched_config),
                "seed": 999,
                "batch_size": 32,
                "steps": 29190,
                "save_freq": 1946,
                "peft": {"r": 16},
            }
        ),
        encoding="utf-8",
    )
    trace_before_config = trace.read_text(encoding="utf-8")
    config_result = _run(
        script,
        ["resume", "--profile", "treatment", "--run-dir", _bash_path(mismatched_config), "--python", _bash_path(tmp_path / "python_stub")],
        env,
    )
    assert config_result.returncode != 0, config_result.stdout + config_result.stderr
    assert trace.read_text(encoding="utf-8").count("train:") == trace_before_config.count("train:")


def test_eval_rejects_non_treatment_and_missing_manifest_references_before_python(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path, real_launcher=False)
    non_treatment = _run(
        script,
        ["eval", "--profile", "no-arrow", "--run-dir", _bash_path(tmp_path / "run"), "--seeds", "1000"],
        env,
    )
    assert non_treatment.returncode != 0
    assert not trace.exists(), "non-treatment eval must be rejected before Python is launched"

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "experiment": "smolvla_lora_treatment_training",
                "base_policy": str(tmp_path / "missing-base"),
                "treatment_adapter": {"path": str(tmp_path / "missing-adapter" / "adapter_model.safetensors")},
            }
        ),
        encoding="utf-8",
    )
    _executable(
        tmp_path / "run_lora_2x2_eval.py",
        f"#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\nwith Path({_bash_path(trace)!r}).open('a', encoding='utf-8') as handle: handle.write('eval:' + json.dumps(sys.argv[1:]) + '\\n')\n",
    )
    missing_refs = _run(
        script,
        ["eval", "--profile", "treatment", "--run-dir", _bash_path(run_dir), "--seeds", "1000", "--python", "/usr/bin/python3"],
        env,
    )
    assert missing_refs.returncode != 0, missing_refs.stdout + missing_refs.stderr
    assert not trace.exists(), "eval must verify manifest-referenced base and adapter before invocation"

    base = tmp_path / "valid-base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "weights.bin").write_bytes(b"immutable weights")
    base_files = {
        "config.json": hashlib.sha256((base / "config.json").read_bytes()).hexdigest(),
        "weights.bin": hashlib.sha256((base / "weights.bin").read_bytes()).hexdigest(),
    }
    (base / "base_snapshot_manifest.json").write_text(
        json.dumps({"revision": "rev-test", "files": base_files}), encoding="utf-8"
    )
    adapter = tmp_path / "valid-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    valid_run = tmp_path / "valid-run"
    valid_run.mkdir()
    (valid_run / "training_manifest.json").write_text(
        json.dumps(
            {
                "experiment": "smolvla_lora_treatment_training",
                "training_variant": "treatment",
                "base_policy": _bash_path(base),
                "base_policy_revision": "rev-test",
                "treatment_adapter": {"path": _bash_path(adapter / "adapter_model.safetensors")},
            }
        ),
        encoding="utf-8",
    )
    valid_eval = _run(
        script,
        ["eval", "--profile", "treatment", "--run-dir", _bash_path(valid_run), "--seeds", "1000", "--python", "/usr/bin/python3"],
        env,
    )
    assert valid_eval.returncode == 0, valid_eval.stdout + valid_eval.stderr
    eval_lines = trace.read_text(encoding="utf-8").splitlines()
    assert any('"--no-videos"' in line and '"--max-videos", "1"' in line for line in eval_lines if line.startswith("eval:"))

    before_mutation = len(eval_lines)
    (base / "weights.bin").write_bytes(b"mutated weights")
    mutated = _run(
        script,
        ["eval", "--profile", "treatment", "--run-dir", _bash_path(valid_run), "--seeds", "1000", "--python", "/usr/bin/python3"],
        env,
    )
    assert mutated.returncode != 0, mutated.stdout + mutated.stderr
    assert len(trace.read_text(encoding="utf-8").splitlines()) == before_mutation

    zero_videos = _run(
        script,
        ["eval", "--profile", "treatment", "--run-dir", _bash_path(valid_run), "--seeds", "1000", "--videos", "--max-videos", "0"],
        env,
    )
    assert zero_videos.returncode != 0


def test_relative_paths_are_normalized_from_an_alternate_working_directory(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path)
    alternate = tmp_path / "operator-cwd"
    alternate.mkdir()
    result = subprocess.run(
        ["bash", _bash_path(script), "dry", "--profile", "treatment", "--run-dir", "relative-run", "--python", _bash_path(tmp_path / "python_stub")],
        cwd=alternate,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (alternate / "relative-run").exists()
    assert "dry run complete" in result.stdout


def test_python_symlink_is_preserved_for_default_and_explicit_interpreters(tmp_path: Path) -> None:
    """Path normalization must not replace a venv's executable symlink."""
    script, env, _ = _runtime(tmp_path)
    env.pop("LAMBDA_VENV", None)
    marker = tmp_path / "python-invocations.log"
    target = tmp_path / "python-target"
    fake_site = tmp_path / "fake_site"
    _executable(
        target,
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$0\" >> {shlex.quote(_bash_path(marker))}\nPYTHONPATH={shlex.quote(_bash_path(fake_site))}${{PYTHONPATH:+:$PYTHONPATH}} exec /usr/bin/python3 \"$@\"\n",
    )

    def make_link(link: Path) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"ln -s -- {shlex.quote(_bash_path(target))} {shlex.quote(_bash_path(link))}",
            ],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    default_link = tmp_path / ".venv-lora" / "bin" / "python"
    explicit_link = tmp_path / "explicit-venv" / "bin" / "python"
    make_link(default_link)
    make_link(explicit_link)

    default_result = _run(
        script,
        ["smoke", "--profile", "treatment", "--run-dir", _bash_path(tmp_path / "default-run")],
        env,
    )
    assert default_result.returncode == 0, default_result.stdout + default_result.stderr

    explicit_result = _run(
        script,
        [
            "smoke",
            "--profile",
            "treatment",
            "--run-dir",
            _bash_path(tmp_path / "explicit-run"),
            "--python",
            _bash_path(explicit_link),
        ],
        env,
    )
    assert explicit_result.returncode == 0, explicit_result.stdout + explicit_result.stderr

    invocations = marker.read_text(encoding="utf-8").splitlines()
    assert any(line.endswith("/.venv-lora/bin/python") for line in invocations)
    assert any(line.endswith("/explicit-venv/bin/python") for line in invocations)


def test_interrupted_resume_audits_are_append_only_and_bind_final_manifest(tmp_path: Path) -> None:
    script, env, trace = _runtime(tmp_path)
    revision = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    base_policy = tmp_path / "base_models" / f"smolvla_libero-{revision}"
    env["BASE_POLICY"] = _bash_path(base_policy)
    data_root = tmp_path / "lora_datasets"
    libero_dir = tmp_path / "LIBERO"
    env["LIBERO_DIR"] = _bash_path(libero_dir)
    libero_commit = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    pair_hash = hashlib.sha256((data_root / "sealed_lora_pair_manifest.json").read_bytes()).hexdigest()
    run_dir = tmp_path / "audit-chain"

    plan = {
        "training_variant": "treatment",
        "dataset_variant": "treatment",
        "dataset_repo_id": "local/libero_spatial_treatment",
        "base_policy": _bash_path(base_policy),
        "base_policy_revision": revision,
        "data_root": _bash_path(data_root),
        "output_dir": _bash_path(run_dir),
        "pair_manifest_sha256": pair_hash,
        "libero_dir": _bash_path(libero_dir),
        "libero_commit": libero_commit,
        "libero_worktree_status": "clean",
        "libero_tracked_clean": True,
        "flags": {"seed": 1000, "peft_r": 16, "batch_size": 32, "steps": 29190, "save_freq": 1946},
    }
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "expected_adapter_inventory.json").write_text(
        json.dumps(_expected_inventory_fixture(_bash_path(base_policy))), encoding="utf-8"
    )
    (run_dir / "training_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run_dir / "run_provenance.json").write_text(
        json.dumps(
            {
                "variant": "treatment",
                "dataset_variant": "treatment",
                "dataset_repo_id": "local/libero_spatial_treatment",
                "base_policy": _bash_path(base_policy),
                "base_policy_revision": revision,
                "libero_dir": _bash_path(libero_dir),
                "libero_commit": libero_commit,
                "libero_worktree_status": "clean",
                "libero_tracked_clean": True,
                "pair_manifest_sha256": pair_hash,
                "flags": {"seed": 1000, "peft_r": 16, "batch_size": 32, "steps": 29190, "save_freq": 1946},
            }
        ),
        encoding="utf-8",
    )

    def config_for(step: int, *, seed: int = 1000) -> Path:
        config = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model" / "train_config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        state = config.parent.parent / "training_state"
        state.mkdir(parents=True, exist_ok=True)
        (config.parent / "adapter_model.safetensors").write_bytes(b"adapter")
        (config.parent / "adapter_config.json").write_text("{}", encoding="utf-8")
        (config.parent / "adapter_audit.json").write_text(
            json.dumps({"expected_inventory_sha256": "fake-inventory"}), encoding="utf-8"
        )
        (state / "optimizer.pt").write_bytes(b"optimizer")
        (state / "scheduler.pt").write_bytes(b"scheduler")
        (state / "random_states_0.pkl").write_bytes(b"rng")
        config.write_text(
            json.dumps(
                {
                    "dataset": {"repo_id": "local/libero_spatial_treatment", "root": _bash_path(data_root / "treatment")},
                    "output_dir": _bash_path(run_dir),
                    "seed": seed,
                    "batch_size": 32,
                    "steps": 29190,
                    "save_freq": 1946,
                    "peft": {"r": 16},
                    "checkpoint_step": step,
                }
            ),
            encoding="utf-8",
        )
        return config

    fail_flag = tmp_path / "resume_fail.flag"
    fail_flag.write_text("fail interrupted attempts", encoding="utf-8")
    step2 = config_for(1946)
    first = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir), "--python", _bash_path(tmp_path / "python_stub")], env)
    assert first.returncode != 0
    audit_dir = run_dir / "resume_audits"
    assert sorted(p.name for p in audit_dir.glob("*.json")) == ["000001.json"]
    first_record = (audit_dir / "000001.json").read_text(encoding="utf-8")

    same = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir), "--resume-config", _bash_path(step2), "--python", _bash_path(tmp_path / "python_stub")], env)
    assert same.returncode != 0
    assert [p.name for p in audit_dir.glob("*.json")] == ["000001.json"]
    assert (audit_dir / "000001.json").read_text(encoding="utf-8") == first_record

    rollback = config_for(1000)
    rollback_result = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir), "--resume-config", _bash_path(rollback), "--python", _bash_path(tmp_path / "python_stub")], env)
    assert rollback_result.returncode != 0
    assert [p.name for p in audit_dir.glob("*.json")] == ["000001.json"]

    step4 = config_for(3892)
    newer = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir), "--resume-config", _bash_path(step4), "--python", _bash_path(tmp_path / "python_stub")], env)
    assert newer.returncode != 0
    assert sorted(p.name for p in audit_dir.glob("*.json")) == ["000001.json", "000002.json"], newer.stdout + newer.stderr
    second_record = json.loads((audit_dir / "000002.json").read_text(encoding="utf-8"))
    first_record_data = json.loads(first_record)
    assert second_record["previous_record_sha256"] == first_record_data["record_sha256"]
    assert second_record["chain_index"] == 2

    fail_flag.unlink()
    final = _run(script, ["resume", "--profile", "treatment", "--run-dir", _bash_path(run_dir), "--resume-config", _bash_path(step4), "--python", _bash_path(tmp_path / "python_stub")], env)
    assert final.returncode == 0, final.stdout + final.stderr
    manifest = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["resume_audits"]) == 2
    expected_chain = hashlib.sha256(
        json.dumps([record["record_sha256"] for record in manifest["resume_audits"]], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert manifest["resume_chain_digest"] == expected_chain
