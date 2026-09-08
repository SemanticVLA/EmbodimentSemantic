# RoboCasa Arrow-Grasp Controller: Complete Handoff

## Purpose

This document records exactly what was attempted, what was implemented, what worked, what failed, what was corrected, and the current state of the RoboCasa integration.

The intended experiment was to evaluate the existing LIBERO arrow-grasp controller on RoboCasa pick-and-place tasks.

The intended evaluation was:

- 21 RoboCasa tasks.
- One episode per task.
- Official RoboCasa evaluation.
- The robot must actually execute arm motion.
- Results must include diagnostics.
- The target was a majority of successful episodes.

No training was involved.

## Scientific premise

The controller algorithm already worked in LIBERO.

The purpose of this work was not to rediscover or redesign the algorithm. The purpose was to determine whether the same algorithm generalizes from LIBERO to RoboCasa.

The RoboCasa implementation was intended to preserve the LIBERO controller logic and change only environment-specific parameters and interfaces.

Permitted RoboCasa-specific changes included:

- Object and fixture nouns.
- Source-object and destination-fixture extraction.
- Camera aliases.
- Image orientation conventions.
- Depth and camera calibration.
- Conversion between RoboCasa world coordinates and the reset-frozen robot-local `B0` frame.
- MuJoCo and RoboCasa API compatibility.
- Packing the canonical LIBERO 7-D action into RoboCasa's official PandaOmron 12-D action.
- RoboCasa reset, stepping, rendering, and official success evaluation.

The following controller behavior was intended to remain logically equivalent to LIBERO:

- MolmoPoint proposal.
- RGB-D geometry.
- Grasp candidate generation.
- Candidate ranking.
- Preshape.
- Approach, grasp, lift, transport, place, and retreat phases.
- Action scales.
- Clearance.
- Timing.
- Retry and recovery behavior.
- Retention gate.
- Official success evaluation after retreat.

A normal grasp or placement failure after actual motion would be valid experimental data. An episode with zero executed actions would indicate an integration failure rather than a controller-performance result.

## Implemented RoboCasa structure

The implementation was organized as a self-contained RoboCasa folder:

```text
vla_benchmarking/robocasa/
  shared/task_manifest.py
  environment/runtime.py
  evaluation/live.py
  evaluation/capture.py
  evaluation/arrow.py
  arrow_grasp_controller/controller/runner.py
  arrow_grasp_controller/controller/episode_contract.py
  arrow_grasp_controller/controller/grasp_candidates.py
  arrow_grasp_controller/controller/molmopoint.py
  arrow_grasp_controller/controller/policy.py
  arrow_grasp_controller/controller/preshape.py
  arrow_grasp_controller/legacy_engine/arrow_controller.py
  arrow_grasp_controller/legacy_engine/rgbd_region.py
  configs/canonical_molmo_rgbd_grasp.json
  configs/active_policy.lock.json
  calibration/
  tests/
  legion/run_robocasa_pickplace.sbatch
```

The RoboCasa production code did not import the LIBERO production package directly.

The `legacy_engine` directory name is historical. Its contents were intended to be the local faithful mirror of the low-level canonical motion engine, not an alternative RoboCasa algorithm.

## Intended high-level controller sequence

The RoboCasa path was structured around the same high-level sequence as LIBERO:

1. Reset the RoboCasa environment.
2. Freeze the reset pose as `B0`.
3. Capture RGB-D and scene state.
4. Render the arrow.
5. Use the RoboCasa-specific noun prompt with MolmoPoint.
6. Generate RGB-D grasp candidates.
7. Rank candidates using the canonical policy.
8. Execute canonical motion phases.
9. Apply retry and recovery behavior.
10. Check object retention.
11. Retreat.
12. Run official RoboCasa success evaluation.

The RoboCasa adapter also handled raw simulator observations, image orientation, camera aliases, camera provenance, robot-local coordinates, PandaOmron action formatting, placement-region validation, structured result writing, and episode archiving.

The raw simulator render was flipped exactly once.

The physical RoboCasa camera alias was `robot0_agentview_left`, while the controller-facing camera identity remained `agentview`.

The canonical 7-D arm/gripper action was packed into the official 12-D PandaOmron action. Base, torso, and mode channels were held at zero.

Placement regions failed closed when no valid interior site was found.

The arrow renderer used the intended OpenCV color `(0, 166, 107)`.

## Local validation that succeeded

The following local syntax check passed:

```text
python -m compileall -q vla_benchmarking/robocasa
```

The RoboCasa test suite passed:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q vla_benchmarking/robocasa/tests -p no:cacheprovider
```

Result:

```text
45 passed, 4 skipped
```

The skipped tests were local OpenCV dependency-gated tests.

A targeted independent probe suite passed:

```text
3 passed, 11 deselected
```

The SLURM script passed shell syntax validation.

The local tests covered task manifests, camera and image provenance, action packing, coordinate transforms, candidate handling, model-name resolution, failure-closed behavior, and controller contracts.

These local tests did not prove that the live compiled RoboCasa MuJoCo model used the same names assumed by the adapter.

Independent review and testing found no remaining high-severity issue in the repository code. That approval was based on local code and local tests, not on a successful live RoboCasa episode.

## Runtime and Legion components that worked

Several Legion jobs successfully passed the expensive environment and asset gates:

- RoboCasa `1.0.1`.
- robosuite `1.5.2`.
- MuJoCo `3.3.1`.
- OpenCV `4.11`.
- Transformers `4.57.1`.
- The pinned MolmoPoint revision.
- A40 GPU runtime.
- BF16 runtime.
- Asset download.
- Environment startup.
- Environment reset.
- Image capture.
- Bounding-box and arrow audit.
- Camera provenance.
- Result writing.
- Archive creation.

The runtime infrastructure was therefore partially functional. The live controller did not successfully execute an arm action in the confirmed canary episodes.

## Legion jobs attempted

Every live canary used:

```text
Task: PickPlaceCounterToStove
Episodes: 1
Seed: 1000
Split: target
GPU: one A40
```

No full 21-task evaluation completed. No majority-success result exists.

### Job 1921849

Commit:

```text
0cecbdb8e81114f44a2959326ff9b858f6fbfba9
```

The runtime and assets initialized successfully.

The job failed before motion with:

```text
property 'body_xpos' of 'MjData' object has no setter
actions_executed = 0
```

The cause was an adapter that shallow-copied MuJoCo data and attempted to assign a read-only `MjData` field.

Archive:

```text
/home/hjaber/EmbodimentSemantic_archive/robocasa/robocasa_arrow_faithful_canary_counter_to_stove_1921849
```

### Jobs 1921896 and 1921907

Both failed at Slurm launch:

```text
JobLaunchFailure
exit 0:53
compute-4-13
```

There was no usable process, log, or run artifact. Later submissions excluded that node.

### Job 1921909

Commit:

```text
310ccdd
```

The runtime started and produced a structured result.

The episode failed before motion with:

```text
model does not expose any site name in
('grip_site', 'gripper0_grip_site', 'robot0_grip_site')
actions_executed = 0
```

The implementation assumed LIBERO/mujoco-py-style name lookup, while the live RoboCasa environment used MuJoCo 3 APIs and different compiled model names.

Archive:

```text
/home/hjaber/EmbodimentSemantic_archive/robocasa/robocasa_arrow_faithful_canary_counter_to_stove_r4_1921909
```

### Job 1921915

Commit:

```text
587b64d
```

The same live gripper-site lookup failure remained.

Archive:

```text
/home/hjaber/EmbodimentSemantic_archive/robocasa/robocasa_arrow_faithful_canary_counter_to_stove_r5_1921915
```

### Job 1921974

Commit:

```text
0643f7e
```

The runtime gate passed and a structured result was written.

The same pre-motion failure remained:

```text
status: controller_failure
terminal_reason: AttributeError
actions_executed: 0
actions_executed_before_motion: 0
official_success: false
error: model does not expose any site name in
('grip_site', 'gripper0_grip_site', 'robot0_grip_site')
```

Run root:

```text
$SCRATCH_FLASH/EmbodimentSemantic_runtime/robocasa/runs/robocasa_arrow_faithful_canary_counter_to_stove_r6_1921974
```

This is the latest confirmed live runtime evidence.

### Latest commit: `a711387`

Commit:

```text
a711387661efc80919b8c628a8539c07d000c753
```

This patch added an authoritative MuJoCo fallback using:

```python
mujoco.mj_id2name(...)
```

for sites, bodies, and geoms when the normal `.name` view was empty.

The patch preserved direct lookup, unique suffix lookup, ambiguity failure, existing geometry behavior, and existing controller parameters.

Local tests passed after this patch.

Execution was stopped before the clean Legion checkout and live submission completed.

Therefore:

- The patch was not verified live.
- It cannot be claimed to have fixed the live failure.
- No current successful RoboCasa job exists.
- No 21-task evaluation has been completed.
- No movement trace has been confirmed.
- No majority-success result exists.

## What was wrong

The following mistakes occurred during the implementation and evaluation process:

1. The initial adapter attempted to assign read-only MuJoCo `MjData` fields.
2. The adapter assumed old mujoco-py/LIBERO `*_name2id` APIs.
3. Local fake tests manually populated `.name` fields in a way that did not match the live compiled model.
4. A live model-introspection probe should have preceded expensive asset setup and Molmo execution.
5. An early RoboCasa version used an unfaithful low-level-only shortcut rather than the full LIBERO-equivalent high-level path.
6. Failures were addressed through sequential patches and repeated job submissions instead of first instrumenting the actual live model/data API.
7. Canary acceptance was not initially hard-gated on actual arm actions and nonzero EEF displacement.
8. Faithfulness to LIBERO was described before live runtime parity had been demonstrated.
9. Paid GPU time was spent before the basic RoboCasa action boundary had been proven.
10. The repeated failures consumed roughly ten hours without producing a successful live RoboCasa episode.

## What was fixed

The following issues were addressed in the repository:

- The shallow-copy/read-only MuJoCo data problem was removed.
- The original low-level-only RoboCasa shortcut was replaced with the full high-level controller path.
- RoboCasa was given its own environment, evaluation, calibration, and controller structure.
- The reset-frozen `B0` coordinate contract was added.
- Camera aliases and image-origin handling were adapted.
- PandaOmron 12-D action packing was added around the canonical 7-D action.
- RoboCasa-specific task nouns and object/fixture extraction were added.
- Placement-region failure handling was made explicit.
- Structured result and archive plumbing was added.
- MuJoCo model-name resolution was expanded to include the official `mujoco.mj_id2name` API.
- Local tests were added for the above behavior.

The latest live evidence still predates the final model-name fallback patch.

## Current state

The repository contains a substantial RoboCasa implementation and a passing local test suite.

The environment and asset setup worked on Legion.

The live controller can reset and capture RoboCasa state.

The confirmed canary episodes failed before arm motion because the runtime could not resolve the expected gripper/end-effector model site.

There is no verified successful RoboCasa episode.

There is no completed 21-task evaluation.

There is no valid majority-success number.

There is no evidence yet that the controller's RoboCasa action adapter produces nonzero EEF motion in the live compiled environment.

## Intended final result

The intended final result remains a self-contained RoboCasa implementation using the same controller logic as LIBERO, with only environment-specific adapters and parameters changed, actual robot motion, one evaluated episode for each of the 21 tasks, official success results, per-episode diagnostics, and a majority-task success rate supported by archived raw episode records.

The integration is currently incomplete.
