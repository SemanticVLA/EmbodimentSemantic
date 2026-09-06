`run_robocasa_pickplace.sbatch` is the isolated Legion launcher for the frozen
21-task, one-episode baseline. It creates or reuses a Python 3.11 environment,
installs the pinned requirements, downloads RoboCasa assets when absent, and
archives the runner outputs and logs. Submit it only from an immutable checkout
with `REPO_ROOT` and `ROBOCASA_EXPECTED_COMMIT` set to the exact commit.
