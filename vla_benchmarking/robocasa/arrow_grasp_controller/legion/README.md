`run_robocasa_pickplace.sbatch` is the isolated Legion launcher for the frozen
RoboCasa Pick & Place controller. It uses one A40 GPU and one SLURM job at a
time, creates or reuses a Python 3.11 environment, installs the pinned
MolmoPoint/RGB-D requirements, downloads RoboCasa assets when absent, and
archives outputs, logs, source/config hashes, and model/cache provenance.
Submit it only from an immutable checkout with `REPO_ROOT` and
`ROBOCASA_EXPECTED_COMMIT` set to the exact commit.

The default is all 21 tasks and one episode per task. Set `ROBOCASA_TASKS` to
`all` or to a comma-separated list of registered task names; the launcher
passes each name as a separate `--tasks` argument without shell evaluation.
For example, a one-task canary uses:

```bash
ROBOCASA_RUN_LABEL=robocasa_canary_cheesybread \
ROBOCASA_TASKS=CheesyBread \
ROBOCASA_EPISODES_PER_TASK=1 \
sbatch vla_benchmarking/robocasa/arrow_grasp_controller/legion/run_robocasa_pickplace.sbatch
```

The 21-task one-episode run uses the defaults (or sets
`ROBOCASA_TASKS=all` and `ROBOCASA_EPISODES_PER_TASK=1` explicitly). The
launcher serializes RoboCasa jobs with a persistent `flock` guard, so a second
submission fails safely while another run is active.
