from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEGION = ROOT / "legion"
BOOTSTRAP = LEGION / "bootstrap_zerograsp.sh"
SETUP = LEGION / "setup_zerograsp_runtime.sh"
SETUP_JOB = LEGION / "setup_zerograsp_runtime.sbatch"
DUAL_MATRIX = LEGION / "run_arrow_pick_place_dual_matrix.sbatch"
PIN = "152f67c27269ff3f089783bd2f041d67641fa506"


def test_bootstrap_is_explicit_pinned_and_never_downloads_checkpoint():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert PIN in text
    assert "https://github.com/sh8/ZeroGrasp.git" in text
    assert "--install-deps" in text
    assert "--verify-only" in text
    assert "--recursive" in text
    assert "https://github.com/TRI-ML/octree_feature_extractor.git" in text
    assert 'git clone --recursive "$ZERO_GRASP_OFFICIAL_URL"' not in text
    assert 'submodule update --init --recursive' in text
    assert "checkout --detach \"$ZERO_GRASP_PIN\"" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "download.sh" not in text
    assert "checkpoint_downloaded=false" in text
    assert "pip install -r \"$ROOT/requirements.txt\"" in text
    assert "PYTHON_LIBERO" in text
    assert '"$ROOT/.venv/bin/python"' not in text
    assert 'VENV_INPUT="${ROOT}.venv"' in text


def test_dual_matrix_zero_grasp_contract_is_opt_in_and_hash_pinned():
    text = DUAL_MATRIX.read_text(encoding="utf-8")
    assert PIN in text
    for variable in (
        "ZERO_GRASP_ROOT",
        "ZERO_GRASP_PYTHON",
        "ZERO_GRASP_PYTHON_SHA256",
        "ZERO_GRASP_CHECKPOINT",
        "ZERO_GRASP_CHECKPOINT_SHA256",
        "ZERO_GRASP_CONFIG",
        "ZERO_GRASP_CONFIG_SHA256",
        "ZERO_GRASP_ENV_LOCK",
        "ZERO_GRASP_ENV_LOCK_SHA256",
    ):
        assert variable in text
    assert "vla_benchmarking/zerograsp_worker.py" in text
    assert "preflight_zerograsp_process.py" in text
    assert "protocol_ready" in text
    assert "process_closed" in text
    assert "zerograsp_process_preflight.json" in text
    assert "--repo \"$ZERO_GRASP_ROOT_RESOLVED\"" in text
    assert 'zerograsp-jsonl-v1' in text
    assert "zerograsp_runtime" in text
    assert "zero_grasp_revision" in text
    assert "zero_grasp_checkpoint_sha256" in text
    assert "zero_grasp_config_sha256" in text
    assert "pip_freeze" in text
    assert "Torch/CUDA identity" in text
    assert '"python_executable_sha256"' in text
    assert '"venv"' in text and '"python"' in text
    assert '"$ZERO_GRASP_PYTHON_RESOLVED" - "$ZERO_GRASP_ENV_LOCK_RESOLVED"' in text

    # The worker handshake must precede the first LIBERO/MuJoCo import and the
    # matrix launcher invocation, which is where environments are constructed.
    handshake = text.index('preflight_zerograsp_process.py')
    libero_import = text.index('import libero, mujoco, robosuite')
    matrix_launch = text.index('"$PYTHON" "$MATRIX_LAUNCHER"')
    assert handshake < libero_import < matrix_launch


def test_dual_matrix_does_not_run_worker_for_legacy_path():
    text = DUAL_MATRIX.read_text(encoding="utf-8")
    selected_block = text[text.index('if [[ "$ZERO_GRASP_SELECTED" == 1 ]]; then'):]
    assert 'if [[ -n "$CONTROLLER_CONFIG_CANONICAL_PATH" ]]' in text
    assert 'ZERO_GRASP_SELECTED="$($PYTHON' in text
    assert 'if [[ "$ZERO_GRASP_SELECTED" == 1 ]]; then' in selected_block
    assert selected_block.count('--handshake') == 0


def test_runtime_setup_matches_official_pinned_compute_contract():
    text = SETUP.read_text(encoding="utf-8")
    for value in (
        PIN,
        "7521c22e2921a0bd8e9285044c842ff6fa2042e0",
        "ae53057eaf36dab01aa2727fcc93a749fd995af5",
        "eb57dd2092d8dbe05312a29c3d0c22f3226efbfc",
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torch-scatter",
        "torch_cluster",
        "xformers==0.0.24",
        "Python 3.11",
        "numpy==1.26.4",
        "numpy_version",
        "build_toolchain",
        "existing runtime lock build_toolchain differs",
        "NVCC_SHA256",
        "CC_SHA256",
        "CXX_SHA256",
        "checkpoint_size_bytes",
        "pip_freeze",
        "--no-deps",
        "octree_feature_extractor_revision",
    ):
        assert value in text
    assert "awk '!/^ocnn[[:space:]]*@/ && !/^dwconv[[:space:]]*@/'" in text
    assert "&& !/^dwconv[[:space:]]*@/" in text
    assert 'DWCONV_URL="git+https://github.com/octree-nn/dwconv.git"' in text
    assert 'OCNN_URL="git+https://github.com/octree-nn/ocnn-pytorch.git"' in text
    assert '"ocnn @ ${OCNN_URL}@${OCNN_PIN}"' in text
    assert '"dwconv @ ${DWCONV_URL}@${DWCONV_PIN}"' in text
    assert "--no-build-isolation --no-deps" in text
    assert "CUDA_HOME with a complete CUDA toolkit and pinned host compilers is required" in text
    assert text.index('torch==2.2.0 torchvision==0.17.0') < text.index('"dwconv @ ${DWCONV_URL}@${DWCONV_PIN}"') < text.index('torch-scatter')
    assert "checkpoint URL is provenance only" in text
    assert "--install" in text and "--smoke" in text and "--verify-only" in text
    assert text.index("if [[ -f \"$LOCK\" ]]; then") < text.index(
        "if ((INSTALL)); then\n  [[ -n \"$CUDA_HOME_CANONICAL\" &&"
    )
    assert "INSTALL=0" in text and "VERIFY_ONLY=1" in text
    assert "zerograsp_setup_mode=acquire" in SETUP_JOB.read_text(encoding="utf-8")
    assert "zerograsp_model_load=false" in SETUP_JOB.read_text(encoding="utf-8")


def test_runtime_setup_job_is_compute_only_and_downloads_only_official_id():
    text = SETUP_JOB.read_text(encoding="utf-8")
    assert "SLURM_JOB_ID" in text
    assert "module load nvhpc-nompi/25.1" in text
    assert "module load gcc/11.5.0" in text
    assert "module load miniforge/24.3.0-0" in text
    assert 'export CUDA_HOME' in text
    assert 'export CUDACXX="$CUDA_HOME/bin/nvcc"' in text
    assert 'export CC CXX CUDAHOSTCXX' in text
    assert 'ZERO_GRASP_CUDA_MODULE_ID="nvhpc-nompi/25.1"' in text
    assert 'ZERO_GRASP_HOST_COMPILER_MODULE_ID="gcc/11.5.0"' in text
    assert "--gres=gpu:1" in text
    assert "gdown==5.2.0" in text
    assert "1xUmFdgT_Ozu4zIPIsh_1SJMcegeQUWqQ" in text
    assert "chmod a-w" in text
    assert "CHECKPOINT_MODE" in text
    assert "checkpoint retains write bits" in text
    assert "zerograsp_process_preflight.json" in text
    assert "--timeout-s 20" in text
    assert "--checkpoint-sha256" in text
    assert 'zerograsp_worker.py" --handshake' not in text
    assert "zerograsp_checkpoint_load=passed" in text
    assert "zerograsp_motion=false" in text
    assert "zerograsp_evaluator=false" in text
    assert text.index('bootstrap_zerograsp.sh" --root') < text.index('CONFIG="${ZERO_GRASP_CONFIG:-${ZERO_GRASP_ROOT}/configs/demo.yaml}"')
    assert "conda create -y -p \"$VENV\" python=3.11 pip" in text
    assert text.index('if [[ -e "$VENV" ]]') < text.index('conda create -y -p "$VENV" python=3.11 pip')
    assert text.index('if [[ -e "$CHECKPOINT" ]]') < text.index("gdown==5.2.0")
    assert "trap archive_on_exit EXIT" in text
    assert "zerograsp-runtime.lock.partial.json" in text
    assert '"$(dirname -- "$ZERO_GRASP_ROOT")"' in text
    assert "git reset" not in text
    assert "rm -rf" not in text
    assert text.index('if [[ "$SETUP_MODE" == "acquire" ]]') < text.index("_load_official_backend")
    assert text.index("zerograsp_model_load=false") < text.index("_load_official_backend")
    assert "execute mode requires explicit ZERO_GRASP_CHECKPOINT_SHA256" in text
    assert "ARROW_MATRIX_EXPECTED_COMMIT" in text
    assert 'SCRIPT_DIR="$(realpath -e -- "$REPO_ROOT/vla_benchmarking/legion")"' in text
    assert "SUBMITTED_SCRIPT_SHA256" in text
    assert "submitted Slurm script differs from the expected evaluation checkout" in text
    assert text.index('if [[ "$SETUP_MODE" == "acquire" ]]') < text.index('WORKER_SCRIPT="$REPO_ROOT/vla_benchmarking/zerograsp_worker.py"')
    assert 'evaluation checkout is dirty' in text
    assert 'setup_script_sha256' in text and 'worker_script_sha256' in text
    assert 'setup_provenance.env' in text
    assert '--checkpoint-sha256 "$CHECKPOINT_SHA"' in text

    # Acquisition is deliberately unable to reach either the process-ready or
    # model-load path. This guards the no-deserialization acquisition split.
    acquire_start = text.index('if [[ "$SETUP_MODE" == "acquire" ]]')
    execute_start = text.index('[[ -f "$ACQUISITION_MANIFEST" ]] || die')
    acquisition_block = text[acquire_start:execute_start]
    assert "preflight_zerograsp_process.py" not in acquisition_block
    assert "_load_official_backend" not in acquisition_block


def test_runtime_lock_rejects_stale_or_different_interpreter_identity():
    setup_text = SETUP.read_text(encoding="utf-8")
    launcher_text = DUAL_MATRIX.read_text(encoding="utf-8")

    # The generated lock must bind the canonical environment and executable,
    # not only a Python major/minor string.
    for field in ("venv", "python", "python_executable_sha256"):
        assert f'"{field}"' in setup_text
    assert "existing runtime lock {key} differs" in setup_text
    assert "python_executable_sha256" in setup_text

    # Matrix preflight runs package/Torch checks under the selected external
    # interpreter and rejects any mismatch before process startup.
    assert '"$ZERO_GRASP_PYTHON_RESOLVED" - "$ZERO_GRASP_ENV_LOCK_RESOLVED"' in launcher_text
    assert "selected interpreter" in launcher_text
    assert "pip_freeze differs" in launcher_text
    assert "Torch/CUDA identity differs" in launcher_text
    assert "NumPy identity differs" in launcher_text
    assert "torchvision identity differs" in launcher_text
    assert '"python_executable_sha256": python_sha' in launcher_text


@pytest.mark.skipif(os.name == "nt", reason="POSIX runtime test requires bash semantics")
def test_execute_mode_without_explicit_checkpoint_hash_fails_before_setup():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this host")
    env = os.environ.copy()
    env.update({
        "SLURM_JOB_ID": "unit-test",
        "REPO_ROOT": str(ROOT),
        "ARROW_MATRIX_EXPECTED_COMMIT": "0" * 40,
    })
    env.pop("ZERO_GRASP_CHECKPOINT_SHA256", None)
    result = subprocess.run([bash, str(SETUP_JOB)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "explicit ZERO_GRASP_CHECKPOINT_SHA256" in result.stderr
    assert "_load_official_backend" not in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="bash syntax check requires POSIX bash")
def test_zerograsp_shell_scripts_are_syntax_valid():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this host")
    for script in (BOOTSTRAP, SETUP, SETUP_JOB, DUAL_MATRIX):
        result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"{script}: {result.stderr}"
