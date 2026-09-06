# RoboCasa PickPlace-21

This package is the self-contained RoboCasa integration for the 21 official
Pick & Place atomic tasks. It owns the task manifest, target-split runtime,
RGB-D capture, bbox projection, arrow rendering, PandaOmron action packing,
runner, outputs, and tests under this folder.

The checked-in arrow motion policy is reused read-only through the local
`arrow_grasp_controller/controller/runner.py` boundary. Its grasp selection,
motion phases, offsets, retry behavior, and action budget are unchanged; this
package supplies only the camera/role/action adaptation around it. No LIBERO
files are modified by this integration.

The first run is a frozen portability baseline on RoboCasa target scenes:

```powershell
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode preflight --output-dir output/robocasa_pickplace21
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode smoke --output-dir output/robocasa_pickplace21_smoke
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint `
  --mode full --execute-motion --output-dir output/robocasa_pickplace21
```

The registered full run is all 21 tasks, ten seeds per task (`1000`–`1009`),
and the `target` scene split.  Every cell is terminal: success, official task
failure, horizon exhaustion, missing geometry, dependency failure, and runtime
exception are all retained in the denominator.

Fixture destinations use task-relevant interior or rack regions.  Arrows are
drawn from source bbox center to destination bbox/region center.  `MakeIcedCoffee`
selects one visible cube deterministically because the official task accepts
either cube.  `PackDessert` is evaluated using RoboCasa's official multi-object
success predicate even though the arrow points only from dessert to its
container.

Calibration is deliberately out of scope for this baseline.  The output keeps
RGB/depth frames, calibration, role bboxes, arrows, prompts, selected points,
action traces, and terminal reasons so a later calibration run can use separate
seeds without rewriting this baseline.
