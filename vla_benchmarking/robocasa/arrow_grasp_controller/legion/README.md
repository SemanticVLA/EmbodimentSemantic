`run_robocasa_pickplace.sbatch` is the isolated Legion launcher for the frozen
RoboCasa Pick & Place controller. It uses one A40 GPU and one SLURM job at a
time, creates or reuses a Python 3.11 environment, installs the pinned
MolmoPoint/RGB-D requirements, downloads RoboCasa assets when absent, and
archives outputs, logs, source/config hashes, and model/cache provenance.
Submit it only from an immutable checkout with `REPO_ROOT` and
`ROBOCASA_EXPECTED_COMMIT` set to the exact commit.

Runtime reuse is the default. With `ROBOCASA_SETUP_RUNTIME=0` (or when the
variable is omitted), the launcher refuses to create or modify the environment:
it requires the existing Python 3.11 environment, checks every package pin,
including Torch 2.7.1 and torchvision 0.22.1, the installed RoboCasa and
robosuite VCS commit in `direct_url.json`, and the
commit-keyed RoboCasa asset marker plus asset directories. Set
`ROBOCASA_SETUP_RUNTIME=1` only for the explicit bootstrap path; that path
creates the environment when absent, installs `requirements.txt`, and resolves
the headless OpenCV provider as before.

The default Transformers cache is `$ROBOCASA_RUNTIME_ROOT/cache/huggingface/transformers`.
Set `ROBOCASA_TRANSFORMERS_CACHE` to an existing absolute cache directory when
the pinned MolmoPoint snapshot is already stored elsewhere; the selected path
is recorded in `runtime_versions.json`, `job_context.env`, and the archived
manifest.

The launcher selects `ROBOCASA_GRASP_PROFILE=canonical_rim` by default. The
explicit `object_contact_v1` profile uses its own checked-in contact config and
is an adapted object-contact treatment; `object_contact_v2` adds the bounded
calibrated camera-ray approach hypothesis while retaining the same canonical
geometry and collision engine. `object_contact_v3` retains both approach
hypotheses, promotes the seed to a bounded observed target component, and
applies target-aware B0 collision filtering while keeping the candidate engine
and motion path unchanged. Neither adapted profile is the canonical rim
policy. The selected profile, config path, and config SHA256 are recorded in
the job context and archive manifest.

After the runtime and asset gates, every job runs the small-motion probe before
the official entrypoint:

```bash
python -m vla_benchmarking.robocasa.tools.probe_runtime \
  --task PickPlaceCounterToStove --seed 1000 \
  --output "$RUN_ROOT/runtime_probe.json" --execute-motion
```

The launcher runs this probe with Hugging Face offline flags, so it cannot
download a model. A nonzero probe exit stops the job before evaluation, and
the probe JSON is copied into the normal archive by the exit trap. The probe
must confirm calibration and at least five executed +Z arm steps of at least
5 mm before the 21-task evaluation can start.

The default is `ROBOCASA_MODE=full`, with all 21 tasks and one episode per
task. For a task subset, the mode defaults to `smoke`. Set
`ROBOCASA_MODE=smoke` or `ROBOCASA_MODE=full` explicitly when desired; full
mode requires `ROBOCASA_TASKS=all` and one episode per task. Set
`ROBOCASA_TASKS` to `all` or to a comma-separated list of registered task
names; the launcher passes each name as a separate `--tasks` argument without
shell evaluation.
For example, a one-task canary uses:

```bash
ROBOCASA_RUN_LABEL=robocasa_canary_cheesybread \
ROBOCASA_TASKS=CheesyBread \
ROBOCASA_EPISODES_PER_TASK=1 \
sbatch vla_benchmarking/robocasa/arrow_grasp_controller/legion/run_robocasa_pickplace.sbatch
```

The 21-task one-episode run uses the defaults (or sets
`ROBOCASA_MODE=full`, `ROBOCASA_TASKS=all`, and
`ROBOCASA_EPISODES_PER_TASK=1` explicitly). The
launcher serializes RoboCasa jobs with a persistent `flock` guard, so a second
submission fails safely while another run is active.
