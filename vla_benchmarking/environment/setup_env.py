"""
One-shot setup for the vla_bench environment.

Requirements:
  - Python 3.12+  (lerobot hard-requires >=3.12)
  - conda or venv already activated
  - lerobot[pi] already installed:
      pip install "lerobot[pi]"

Then run:
    python vla_benchmarking/environment/setup_env.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── sanity-check Python version ───────────────────────────────────────────────
if sys.version_info < (3, 12):
    sys.exit(
        f"ERROR: Python 3.12+ is required (lerobot constraint). "
        f"You are running {sys.version}.\n"
        f"Create a new env: conda create -n vla_bench python=3.12 && conda activate vla_bench"
    )

VLA_ROOT   = Path(__file__).resolve().parents[1]
REPO_ROOT  = VLA_ROOT.parent
# Preserve the historical checkout location while keeping this setup helper
# under its dedicated environment package.
LIBERO_DIR = VLA_ROOT / "LIBERO"
LIBERO_GIT = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"


def run(cmd, **kwargs):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


# ── 1. Clone LIBERO ───────────────────────────────────────────────────────────
step("1. Clone LIBERO source")
if LIBERO_DIR.exists():
    print(f"  Already exists at {LIBERO_DIR}, skipping clone.")
else:
    run(["git", "clone", LIBERO_GIT, str(LIBERO_DIR)])

# ── 2. Install lerobot[pi] (must come before requirements.txt) ────────────────
step("2. Install lerobot[pi]")
run([sys.executable, "-m", "pip", "install", "lerobot[pi]"])

# ── 3. Install remaining pip dependencies ─────────────────────────────────────
step("3. Install pip dependencies from requirements.txt")
req = VLA_ROOT / "requirements.txt"
run([sys.executable, "-m", "pip", "install", "-r", str(req)])

# ── 4. Install LIBERO as editable package ─────────────────────────────────────
step("4. Install LIBERO from local clone")
run([sys.executable, "-m", "pip", "install", "-e", str(LIBERO_DIR)])

# ── 5. Fix LIBERO import path ─────────────────────────────────────────────────
# The editable install maps `libero` -> LIBERO/libero/ (the inner package dir).
# But lerobot does `from libero.libero import ...` which needs LIBERO/ on sys.path
# so that `libero` resolves to LIBERO/libero/ which contains the `libero/` subpackage.
# A .pth file is the cleanest fix.
step("5. Fix libero.libero import path")
site = Path(sys.prefix) / "Lib" / "site-packages"
pth = site / "libero_src.pth"
pth.write_text(str(REPO_ROOT / "LIBERO") + "\n", encoding="utf-8")
print(f"  Written: {pth}")

# Verify
result = subprocess.run(
    [sys.executable, "-c", "from libero.libero import benchmark; print('  libero.libero import OK')"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(result.stdout.strip())
else:
    print(f"  WARNING: libero.libero import still failing:\n{result.stderr}")

# ── 6. Windows-only patches ───────────────────────────────────────────────────
if platform.system() == "Windows":
    step("6. Apply Windows patches for robosuite / mujoco")

    # 6a. robosuite 1.4.x expects its own bundled mujoco.dll inside robosuite/utils/.
    #     It is not shipped in the wheel — copy it from robosuite in the old vla_bench
    #     env if available, or from the mujoco package if present.
    dst_dll = site / "robosuite" / "utils" / "mujoco.dll"
    if dst_dll.exists():
        print(f"  mujoco.dll already in place at {dst_dll}, skipping.")
    else:
        # Try to find it from an installed mujoco package
        src_dll = site / "mujoco" / "mujoco.dll"
        if src_dll.exists():
            shutil.copy2(src_dll, dst_dll)
            print(f"  Copied {src_dll} -> {dst_dll}")
        else:
            print(
                f"  WARNING: mujoco.dll not found at {src_dll}.\n"
                f"  You must manually copy mujoco.dll into:\n"
                f"    {dst_dll}\n"
                f"  Source: from another conda env that has robosuite 1.4.x installed,\n"
                f"  e.g.: <other_env>/Lib/site-packages/robosuite/utils/mujoco.dll"
            )

    # 6b. robosuite forces MUJOCO_GL=egl when MUJOCO_GPU_RENDERING=True,
    #     but egl is Linux-only. Disable GPU rendering so it falls back to glfw/wgl.
    macros_path = site / "robosuite" / "macros.py"
    if macros_path.exists():
        text = macros_path.read_text(encoding="utf-8")
        if "MUJOCO_GPU_RENDERING = True" in text:
            text = text.replace("MUJOCO_GPU_RENDERING = True", "MUJOCO_GPU_RENDERING = False")
            macros_path.write_text(text, encoding="utf-8")
            print(f"  Patched MUJOCO_GPU_RENDERING=False in {macros_path}")
        else:
            print(f"  macros.py already patched or not found with expected value.")
    else:
        print(f"  WARNING: {macros_path} not found — skipping macros patch.")

else:
    step("6. Skipping Windows-only patches (not on Windows)")

# ── 7. Write project-local LIBERO config ──────────────────────────────────────
# lerobot's libero env reads ~/.libero/config.yaml for asset/init paths.
# Writing a project-local one avoids conflicts with other LIBERO installs.
step("7. Write project-local LIBERO config (~/.libero/config.yaml)")
libero_inner = LIBERO_DIR / "libero" / "libero"
user_libero_config = Path.home() / ".libero" / "config.yaml"
user_libero_config.parent.mkdir(exist_ok=True)
config_content = (
    f"assets: {libero_inner / 'assets'}\n"
    f"bddl_files: {libero_inner / 'bddl_files'}\n"
    f"benchmark_root: {libero_inner}\n"
    f"datasets: {libero_inner.parent / 'datasets'}\n"
    f"init_states: {libero_inner / 'init_files'}\n"
)
user_libero_config.write_text(config_content, encoding="utf-8")
print(f"  Written: {user_libero_config}")

# ── 8. Smoke test ─────────────────────────────────────────────────────────────
step("8. Smoke test")
smoke = (
    "from libero.libero import benchmark, get_libero_path; "
    "from libero.libero.envs import OffScreenRenderEnv; "
    "import lerobot; "
    "import robosuite; "
    "print('  OK — all imports successful')"
)
run([sys.executable, "-c", smoke])

print(
    "\nSetup complete.\n"
    "\nNOTE: lerobot/pi0 models are large (~3-8 GB). A CUDA GPU is strongly\n"
    "recommended. On CPU inference will be extremely slow or may OOM.\n"
    "\nRun eval (PowerShell):\n"
    '  $env:CONTEXT_MODE="scene_graph"; $env:TASK_IDS="[3]"; '
    "python vla_benchmarking/evaluation/run_lerobot_eval_with_context.py\n"
    "\nRun eval (bash/zsh):\n"
    "  CONTEXT_MODE=scene_graph TASK_IDS=[3] python vla_benchmarking/evaluation/run_lerobot_eval_with_context.py\n"
)
