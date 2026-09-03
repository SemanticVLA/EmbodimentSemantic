# Active arrow controller

The only executable arrow policy is
`v9d_rgbd_region_grasp_search.json`:

`libero_spatial_akita_bowl_agentview_v9d_rgbd_region_grasp_search`

Its semantic configuration hash is
`60f4f5f9ecfde7b4830f376ab06cfc706e2ef175d86817c42a0adb7cddd46c0c`.

## Runtime contract

The controller receives only the clean RGB image, one-arrow RGB image,
aligned metric depth, camera calibration, and proprioception. The arrow is
decoded from the rendered RGB input. Bboxes and scene-graph data are used
only upstream to render that arrow and record provenance. Object poses,
simulator state, task metadata, and evaluator results are forbidden controller
inputs; evaluation is called only after motion and retreat.

The RGB-D region policy derives bounded source grasp candidates from the
source image/depth region only after a qualifying initial-attempt failure and
tries at most three additional candidates. Motion executes
the explicit phases `pregrasp`, `descend`, `close`, `lift`, `preplace`,
`descend_place`, `open`, and `retreat`. Close/lift stall, timeout, and
empty-gripper signals can trigger the bounded RGB-D grasp-search retry; all
attempts and proprioception are recorded in the audit. Destination release remains
arrow-derived and is checked against the RGB-D/workspace contract.

Every episode records the selected policy name/hash, capture/depth contract,
phase records, motion status, retries, frames/videos when enabled, and the
terminal evaluator result. Use the matrix launcher with no controller
override to select v9d:

```text
python -m vla_benchmarking.run_arrow_pick_place_matrix --execute-motion --allow-unvalidated-profile
```

To select the policy explicitly, pass
`--controller-config vla_benchmarking/controller_configs/v9d_rgbd_region_grasp_search.json`.
Retired arrow and ZeroGrasp policies are rejected before environment
construction. Fine-tuned VLA/LoRA configurations are separate and remain
available.

## Episode flow, end to end

1. **Resolve the policy.** The direct runner and matrix runner load this file
   before constructing LIBERO. They expand and hash the JSON, record the
   source path and semantic hash, and reject any retired name, retired file,
   or modified same-name payload. Standard Legion launchers also reject
   ambient controller/config overrides.

2. **Construct and settle the scene.** LIBERO is created with the `agentview`
   RGB-D camera and the selected suite mode (`vanilla` or
   `sealed_randomized`). The environment is settled before capture/motion;
   settling diagnostics are part of the episode record.

3. **Capture the controller inputs.** One aligned 256x256 clean RGB frame and
   depth frame are captured from the same render call. Depth is declared as
   normalized or metric, sanitized, converted to meters when needed, and
   audited for finite, positive endpoint support. Camera intrinsics and the
   world-from-camera transform are recorded. RGB/depth shape and alignment are
   fail-closed checks.

4. **Render and decode the arrow.** Upstream task metadata may identify the
   bowl and plate only to draw exactly one subject-to-goal arrow on a copy of
   the clean RGB. The controller receives the clean RGB, the resulting
   one-arrow RGB, metric depth, calibration, and proprioception—not the
   bounding boxes or scene graph. The arrow tail/head pixels are decoded from
   the image difference and checked against depth.

5. **Build source and destination geometry.** Arrow endpoint depth patches are
   deprojected through the camera calibration into world coordinates. The
   first attempt uses the fixed, audited endpoint offsets
   `source_grasp = source_visual_endpoint + (0.0146, 0.0432, 0.0244) m` and
   `destination_release = destination_visual_endpoint + (-0.0057, 0.0484,
   0.0310) m`. The active policy uses lower-quantile endpoint depth
   (`q=0.25`); it does not use object poses. Destination release stays
   arrow-derived and is checked against the RGB-D/workspace contract.

6. **Execute the fixed phases.** With OSC positional control, the runner
   executes `pregrasp → descend → close → lift → preplace →
   descend_place → open → retreat`. Each phase has an explicit tolerance and
   bounded action budget. If the initial fixed-offset attempt hits a configured
   close/lift stall, timeout, or empty-gripper trigger, the runner then derives
   a bounded RGB-D source-region candidate list from the original capture and
   retries up to three additional candidates. Candidates are filtered by
   valid depth, region support, mask fraction, and workspace bounds; no
   evaluator query is made during motion.

7. **Audit and evaluate.** Every phase records target, residuals, EEF state,
   gripper state, action counts, and failure/timeout information. Captures,
   phase frames, motion traces, retry records, hashes, and provenance are
   written to the run output (and archived by the launcher). Only after the
   complete motion and retreat sequence does the runner query the LIBERO
   evaluator and append the terminal success/failure record.

## What the hash means

The semantic hash identifies the expanded v9d policy above. Runtime manifests
also include the controller source hashes, suite mode, capture/depth contract,
commit, seed, task, and resolution. These identities make a result comparable
to the frozen v9d canary/200-cell runs without silently mixing in a retired
variant. Historical JSON, videos, and ZeroGrasp outputs remain in their
archives; they are not executable defaults.
