# Canonical grasp controller internals

This package is the implementation behind the single public entrypoint
`vla_benchmarking.libero.arrow_grasp_controller.run_grasp_controller`. It is intentionally modular, but its
modules are not independent operator policies:

- `runner.py` owns capture provenance, candidate ordering, bounded retries, and
  the episode adapter;
- `molmopoint.py` owns the pinned MolmoPoint proposal worker;
- `grasp_candidates.py` owns RGB-D support checks and executable grasp
  geometry;
- `preshape.py` owns measured gripper opening/preshape behavior.

The Legion operator surface is the one launcher
`vla_benchmarking/libero/arrow_grasp_controller/legion/run_grasp_controller.sbatch`.
Do not invoke these modules directly for a new evaluation.

## Immutable default contract

The package preserves the verified `failure_opening40_retreat80` behavior from
release `b4fb87759ae3a1ea2cd518cd201a1a737bb14e80`:

1. Capture aligned agentview RGB-D and the existing calibration together.
2. Use the arrow-derived bowl support and MolmoPoint's clearance-aware rim
   proposals; do not use SAM.
3. Expand valid proposals into executable jaw positions, yaw, insertion,
   opening, approach, release, and retreat poses using RGB-D geometry.
4. Rank deterministically and execute at most four attempts. After an empty
   close, close/lift timeout, or failed retention indication, open, retreat,
   capture a fresh frame, regenerate candidates, and select a different one.
5. Apply the measured 40 mm preshape, 20 mm release-height compensation, and
   80 mm post-release retreat while preserving the existing phase and action
   limits.
6. Query the evaluator only after placement and retreat.

Model revision, prompt identity, camera/depth contract, calibration, release
identity, and candidate/retry records are written to the run manifest. Missing
or inconsistent provenance fails closed. The runner must not consume
simulator object poses or evaluator results to choose a grasp.

## Frozen evidence and rollback

The behavior-equivalent frozen release produced **87/100 (87%)** on
sealed-randomized Legion job `1920556`. The raw output archive is:

```text
/home/hjaber/EmbodimentSemantic_archive/molmo_failure_sealed100/
molmo_failure_sealed100_fa1ae83_1920556
```

Keep Git history and Legion archives as the rollback/audit record. A behavior
change requires a new experiment identity and a new release; it must not be
introduced as a hidden variant of this package.

The 87% result is evidence for the executed release above. A later
behavior-equivalent refactor must pass its own smoke/regression checks before
being described as separately evaluated.

## How the controller works

For a longer plain-language explanation with the mathematical basis, see
[the expanded walkthrough](HOW_IT_WORKS_EXPLAINED.md) or its
[LaTeX version](HOW_IT_WORKS_EXPLAINED.tex).

### 1. Entry point and scope

- This describes the current canonical LIBERO Panda controller.
- Start at `run_grasp_controller.py`. It calls `controller/entrypoint.py`, then `controller/runner.py`.
- The runner uses the locked `canonical_molmo_rgbd_grasp` policy.
- It uses MolmoPoint for grasp proposals and `OSC_POSE` for robot motion.
- The grasp prompt and geometry assume a bowl rim.
- It does not yet choose a triplet from a user query.
- Its public command creates arrows internally. It has no arbitrary-arrow-image command option.

### 2. Scene setup and robot preparation

- The matrix runner creates and settles each LIBERO scene.
- The canonical run uses the fixed `agentview` camera at 256 × 256 pixels.
- It also creates the wrist camera. Grasp proposals use `agentview`.
- The robot stays at its parked observation pose. There is no extra observation-hover move.
- The gripper first opens for 20 control steps while holding its pose.
- A robot calibration probe measures the finger pads, jaw axis, opening, and grip-site frame.
- The grip site is the robot control point. It is offset from the finger contact point.
- The gripper then closes in small pulses toward a nominal 40 mm opening.
- It accepts a measured opening of 35–45 mm. Three readings must agree within 0.25 mm.
- It takes a fresh RGB-D image after this preparation.

### 3. Arrow creation

- The input builder gets live object boxes from simulator geometry.
- The scene graph uses `SCENE_GRAPH_SUBJECT_FILTER = None`.
- The arrow source is separately set by `ARROW_SOURCE_OBJECT = "akita_black_bowl_1"`.
- The destination comes from `TASK_GOAL_OBJECT_CONFIG`. It is currently `plate_1` for all ten tasks.
- The renderer draws one green arrow from the bowl-box center to the destination-box center.
- It creates a synthetic `(source, "goal", destination)` edge. The relation label does not control motion.
- It draws on a copy of the clean RGB image. Both images keep the same pixel frame.
- For spans below 32 pixels, it reduces the line width and arrowhead size.
- Boxes and object names choose the input arrow. They do not choose the physical rim grasp.
- The control path uses the clean RGB, arrow RGB, aligned metric depth, camera calibration, and robot state.

### 4. Localization from the arrow

- The decoder subtracts the clean image from the arrow image.
- It groups changed pixels and requires one clear arrow component.
- It finds the arrow's main axis with PCA, using SVD.
- It compares the shape at both ends to find the arrowhead.
- The tail gives the source pixel `(u_s, v_s)`.
- The head gives the destination pixel `(u_t, v_t)`.
- Missing, very short, or ambiguous arrows cause an error.
- The decoder reads arrow geometry. It does not ask MolmoPoint to read the arrow.

### 5. Localizing the bowl region

- The tail pixel seeds a region in the depth image.
- The code takes the median valid depth in a 7 × 7 patch around the tail.
- It converts a 75 mm search radius into pixels at that depth.
- It keeps pixels within 25 mm of the seed depth.
- It grows the connected region from the nearest valid seed pixel.
- The region must contain at least 48 pixels and at most 15% of the image.
- This region is an observed depth mask. It is not a full object mesh.
- SAM is not used. The `sam_mask` argument name in the grasp code is historical.

### 6. Finding grasp points

- The code highlights the depth mask in red on the clean RGB image.
- It sends that image to the pinned `allenai/MolmoPoint-8B` model.
- The prompt asks for exposed bowl-rim points that fingers can reach without hitting nearby objects.
- The model returns image points. It does not return robot actions or complete grasp poses.
- The code checks that every returned point is finite and inside the original image.
- Depth and camera calibration convert the observed bowl pixels into world points.
- The code removes points inside the measured robot-hand volume.
- It estimates the upper rim from the 90th percentile of world Z, with an 8 mm lower band.
- Molmo points snap to this upper-rim support within 12 pixels.
- The `molmo_dense` policy adds evenly spread support points, up to 16 total seeds.

### 7. Building and choosing a grasp

- Each seed uses nearby rim support within 15 mm in 3D.
- Local PCA estimates the rim tangent.
- The approach points downward. The jaw crosses the rim tangent.
- The code tests yaw offsets of −15°, 0°, and +15°.
- It tests insertion depths of 0, 4, and 8 mm along the downward approach.
- Required opening is the local width across the jaw plus 4 mm clearance per finger.
- The robot calibration converts each contact point into a grip-site position and rotation.
- The pregrasp point is 80 mm above that grip-site position.
- Candidates must fit the measured opening and the configured workspace.
- A depth-based check tests the hand volume along the pregrasp-to-contact path.
- It requires 6 mm obstacle clearance. It allows local target contact near the final grasp.
- It also tests the equivalent 180° jaw flip and prefers the feasible frame with less rotation.
- Ranking uses Molmo proximity, opening margin, depth support, insertion, and robot movement.
- Near-duplicate poses are removed. At most 128 candidates remain.
- The runner executes the highest-ranked candidate whose ID has not already failed.

### 8. Movement from the arrow: exact target calculation

- The two arrow endpoints determine a 3D transfer displacement.
- Each endpoint uses the 25th percentile of valid depth in a 5 × 5 patch.
- Pixel coordinates use `u` to the right and `v` downward.
- The camera matrix is adjusted to match that image convention.
- Each endpoint becomes a camera point: `p_camera = ((u-cx)*d/fx, (v-cy)*d/fy, d)`.
- The camera-to-world transform gives `p_world = R * p_camera + t`.
- Depth and calibration set the distance in metres. There is no fixed pixel-to-metre scale.
- Let `S` be the tail's world point and `T` the head's world point.
- Let `G` be the chosen rim grasp's grip-site position. Usually, `G` is not `S`.
- The active policy is `legacy_displacement`. It includes two fixed world-frame offsets.
- Source offset: `o_s = (0.0146, 0.0432, 0.0244)` metres.
- Destination offset: `o_t = (-0.0057, 0.0484, 0.0310)` metres.
- Transfer displacement: `delta = (T + o_t) - (S + o_s)`.
- Equivalently: `delta = T - S + (-0.0203, 0.0052, 0.0066)` metres.
- The nominal destination for the grip site is `P = G + delta`.
- The final release target is `P + (0, 0, 0.020)` metres.
- The source offset is not added to `G`. The two offsets affect the transfer displacement only.
- The arrowhead therefore does not directly become the final gripper position.
- Grasp geometry sets the gripper rotation. The arrow does not set yaw or object rotation.
- The robot follows staged waypoints. It does not trace the drawn line in the camera image.

### 9. Grasping: physical motion order

- `vertical_clearance`: move vertically above the current hand position.
- Its height is `max(current_hand_z, pregrasp_z)`.
- Hold the initial hand rotation during this first move.
- `rotate`: turn to the chosen grasp rotation at that height.
- `translate_clearance`: move across to the pregrasp X/Y at the same height.
- `pregrasp`: move to the candidate's pregrasp point.
- `descend`: move down to `G`.
- `close`: hold the grasp pose and close for 20 control steps.
- If `max(abs(gripper_qpos)) <= 0.0015`, treat the close as likely empty.
- `lift`: rise to the shared transfer height, `max(G.z, P.z) + 0.080` metres.
- After lifting, require finite joint readings and `max(abs(gripper_qpos)) >= 0.0015`.
- These joint-position checks are contact hints, not visual proof of a secure grasp.

### 10. Carrying, release, and retreat

- `preplace`: move to `P.x, P.y` at the shared transfer height.
- This keeps the main carry move horizontal in world coordinates.
- `descend_place`: lower to `P.z + 0.020` metres.
- `open`: hold this release pose and open for 20 control steps.
- `retreat`: move vertically 80 mm above the release pose.
- The selected rotation is first commanded at `rotate`. Hold it through all later motion phases.
- The current policy uses the depth-based destination and fixed offsets. It does not fit a destination support plane.
- The grasp approach has an obstacle check. The complete carry path has no general collision planner.

### 11. How each waypoint becomes robot actions

- At every control step, read the current hand position and rotation.
- Compute the error from the current pose to the active waypoint.
- Send seven values: three position commands, three rotation commands, and one gripper command.
- Divide position error by 0.05 m and rotation-vector error by 0.5 rad.
- Clip each command to `[-1, 1]` and send it to `OSC_POSE`.
- Gripper commands are `+1` to close, `-1` to open, and `0` during other phases.
- Use robot calibration to align the reported hand rotation with the grip-site target rotation.
- Most move phases accept a 15 mm position error. Final retreat accepts 5 mm.
- Phases with a rotation target also require at most 0.12 rad rotation error.
- Each move phase has a 160-step limit. One shared 1,200-action budget covers preparation, attempts, and recovery.
- Targets stay fixed during an attempt. Pose feedback drives the hand toward them; visual targets refresh between attempts.

### 12. Failure, retry, and result

- An empty close, failed lift check, or motion error can trigger recovery.
- Recovery opens the gripper and retreats 100 mm upward from its current position.
- A failed recovery stops further attempts.
- After successful recovery, repeat the measured preshape and capture fresh RGB-D.
- Rebuild and decode the arrow. Refresh robot calibration and regenerate grasp candidates.
- Exclude previously failed candidate IDs. Execute one candidate per fresh capture.
- Allow at most four grasp attempts. Stop if no valid candidate remains or the shared budget runs out.
- Query the task evaluator only after placement and completed retreat.
- A negative task score does not itself trigger another grasp attempt.
- Save images, decoded endpoints, candidate records, motion traces, failure reasons, and result manifests.

### Source files and verification

- [entrypoint.py](controller/entrypoint.py): public run settings and active policy selection.
- [canonical_molmo_rgbd_grasp.json](configs/canonical_molmo_rgbd_grasp.json): exact model, geometry, motion, and retry constants.
- [runner.py](controller/runner.py): preparation, arrow refresh, perception, retries, and evaluator gating.
- [run_arrow_pick_place_matrix.py](../evaluation/run_arrow_pick_place_matrix.py): scene input and explicit arrow source.
- [molmopoint.py](controller/molmopoint.py): model input and point decoding.
- [grasp_candidates.py](controller/grasp_candidates.py): rim support, pose geometry, collision checks, and ranking.
- [preshape.py](controller/preshape.py): measured gripper opening.
- [arrow_controller.py](legacy_engine/arrow_controller.py): arrow decoding, depth-region extraction, deprojection, and waypoint helpers.
- [run_arrow_pick_place_eval.py](../evaluation/run_arrow_pick_place_eval.py): endpoint depth, transfer equations, release offsets, and ordered motion.
- Verified against the current source code on 2026-09-06. This document does not report a new live simulation result.
