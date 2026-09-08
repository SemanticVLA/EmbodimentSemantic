# RoboCasa PickPlace-21

This package is the self-contained RoboCasa integration for the 21 official
Pick & Place atomic tasks. It owns the task manifest, target-split runtime,
RGB-D capture, bbox projection, arrow rendering, PandaOmron action packing,
runner, outputs, and tests under this folder.

The checked-in arrow motion policy is reused through the local
`arrow_grasp_controller/controller/runner.py` boundary. The default
`canonical_rim` grasp profile preserves the LIBERO bowl-rim pipeline. The
separate `object_contact_v1` profile changes the prompt and observed contact
support for other objects while retaining the canonical candidate geometry,
collision checks, motion phases, offsets, retry behavior, and action budget.
The exploratory `object_contact_v2` profile keeps that same object-contact
proposal and canonical engine, then evaluates a calibrated camera-to-contact
approach alongside world-down, with strict original-frame workspace and
collision checks. It adds no training or simulator-state inputs.
The `object_contact_v3` treatment keeps both approach hypotheses but promotes
the admitted RGB-D seed to a bounded observed upper-surface component. It then
runs the unchanged candidate geometry against a RoboCasa-local B0 sweep that
keeps all observed points in collision checking and permits only late contact
inside the measured finger/pad capture volume. This lets a solid cube or
cheese sit between the fingers while counters, walls, and fixtures remain
obstacles. Robot-occluded arrow tails fail closed instead of using foreground
robot depth. This is still zero-shot and does not alter the LIBERO engine.
No LIBERO files are modified by this integration.

The first run is a frozen portability baseline on RoboCasa target scenes:

```powershell
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode preflight --output-dir output/robocasa_pickplace21_preflight
python -m vla_benchmarking.robocasa.tools.probe_runtime `
  --task PickPlaceCounterToStove --seed 1000 --execute-motion `
  --output output/robocasa_runtime_probe.json
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode smoke --execute-motion --tasks PickPlaceCounterToStove `
  --episodes-per-task 1 --output-dir output/robocasa_pickplace21_smoke
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode full --execute-motion --episodes-per-task 1 `
  --output-dir output/robocasa_pickplace21
```

The requested full run is all 21 tasks, one seed per task (`1000`),
and the `target` scene split.  Every cell is terminal: success, official task
failure, horizon exhaustion, missing geometry, dependency failure, and runtime
exception are all retained in the denominator.

Fixture destinations use task-relevant interior or rack regions.  Arrows are
drawn from source bbox center to destination bbox/region center.  `MakeIcedCoffee`
selects one visible cube deterministically because the official task accepts
either cube.  `PackDessert` is evaluated using RoboCasa's official multi-object
success predicate even though the arrow points only from dessert to its
container.

Policy tuning is deliberately out of scope for this baseline. Robot-frame and
action-interface verification are required before evaluation. The output keeps
RGB/depth frames, calibration, role bboxes, arrows, prompts, selected points,
action traces, and terminal reasons so a later calibration run can use separate
seeds without rewriting this baseline.

The runtime probe uses its own unscored environment. Require it to pass before
the scored canary; inspect actual arm motion and official outcome before the
full matrix. The [Legion launcher](arrow_grasp_controller/legion/README.md)
enforces the runtime gate and records the immutable source and runtime.
Completed cells are saved after every episode; an interrupted matrix retains
its planned denominator and can resume the missing cells with the same identity.

See [repair evidence](REPAIR_RUN_RECORD.md) for the diagnosed failures and live
validation records.

The original-policy 21-task run completed with 0/21 successes. Follow-up
inspection identified source-contact errors and a moving-base coordinate bug.
The corrected adapter freezes perception/proprioception in the reset frame B0
and transforms each action into the current OSC base frame. The next exploratory
pass uses the separately identified object-contact profile:

```powershell
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode full --execute-motion --tasks all --episodes-per-task 1 `
  --seed-base 1000 --split target --grasp-profile object_contact_v1 `
  --output-dir output/robocasa_object_contact_v1
```

Use a new output directory for a changed source or profile; incompatible resume
identities are rejected to preserve prior results. This development pass combines
the frame correction and grasp adaptation, so it cannot isolate their effects
or establish held-out generalization. The completed live stages shared one
existing SLURM allocation, following the user's instruction.

The next bounded treatment uses the same one-allocation 21-task protocol with
the reviewed access-cone profile:

```powershell
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode full --execute-motion --tasks all --episodes-per-task 1 `
  --seed-base 1000 --split target --grasp-profile object_contact_v2 `
  --output-dir output/robocasa_object_contact_v2
```

The current exploratory full-matrix treatment is `object_contact_v3`:

```powershell
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode full --execute-motion --tasks all --episodes-per-task 1 `
  --seed-base 1000 --split target --grasp-profile object_contact_v3 `
  --output-dir output/robocasa_object_contact_v3
```
