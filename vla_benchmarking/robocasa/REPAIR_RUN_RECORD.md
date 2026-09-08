# RoboCasa experiment repair — 2026-09-06

This record continues `ROBOCASA_GPT6_HANDOFF.md`; historical failed runs are
preserved. It records integration evidence separately from scored evaluation.

**Execution resumed on 2026-09-06 and completed.** The original diagnostic job
`1922154` completed cleanly at `21:06:25 UTC`; the adapted run then completed
in job `1922186` at `21:35:38 UTC`, also with exit `0:0`. The reviewed source is
`e4d9683a110944cf95719423266aa5e5c37a38e8`. The adapted 21-task result is
complete and preserved below. The next section is the follow-up research and
repair plan; no additional live run is queued.

## Original frozen evaluation contract

- Question: does the existing LIBERO arrow-grasp controller transfer to RoboCasa?
- Protocol: the 21 registered Pick & Place tasks, one episode each, seed 1000,
  official target scenes and official task success predicates.
- No training or target-task policy tuning. Preserve the canonical policy lock,
  MolmoPoint revision, grasp generation/ranking, phases, action scales, timing,
  retry/recovery, retention gate, and evaluation after retreat.
- Allowed changes: RoboCasa environment/API, robot/camera/frame adapters,
  diagnostics, runtime gates, and faithful result accounting.
- All 21 cells remain in the denominator. Majority means at least 11 official
  successes. A functioning integration alone does not establish generalization.
- One episode per task provides descriptive results, not a precise estimate of
  performance over seeds or layouts. Debugging on seed 1000 is disclosed here.

## Baseline diagnosis (completed)

- Source: unchanged handoff commit
  `a711387661efc80919b8c628a8539c07d000c753`.
- Local tests before repair: 45 passed, 4 OpenCV-gated skips.
- Live diagnostic SLURM job: `1922110`, one A40, 15 seconds elapsed,
  completed with exit `0:0`. This diagnostic intentionally caught calibration
  failure so it could test the action boundary independently; its process exit
  is not a calibration pass.
- Task: `PickPlaceCounterToStove`, seed 1000, separate unscored environment.
- Model/data: `robosuite.utils.binding_utils.MjModel` / `MjData`.
- Authoritative right EEF: site ID 3, `gripper0_right_grip_site`, confirmed by
  both `robot.eef_site_id` and `gripper.important_sites`.
- Calibration still failed on the latest handoff commit. A callable legacy
  name lookup raised before the new suffix fallback could execute. Its
  MuJoCo fallback also needed the raw wrapped model.
- Live rotation arrays use `(N, 9)` storage. The previous adapter transformed
  positions to B0 while leaving these rotations in world coordinates.
- Ten canonical actions `[0, 0, 0.1, 0, 0, 0, -1]` through the existing 12-D
  adapter moved the EEF by `0.010280473957644904 m` (positive Z). This verifies
  the action boundary, not a grasp or task success.
- Raw diagnostic:
  `/home/hjaber/EmbodimentSemantic_archive/robocasa/api_probe_gpt6_1922110/probe.json`.
- Diagnostic source:
  `/home/hjaber/EmbodimentSemantic_runtime/operator/jobs/robocasa_api_probe_gpt6.py`.

## Repair validation

- Calibration-only immutable source:
  `d8dbc58ed3cbf87612756184a9db4fb2faa57347` (parent: handoff commit).
- Live diagnostic job `1922134`: calibration passed and all ten arm commands
  executed. Site ID 3 resolved to `gripper0_right_grip_site`; hand body ID 16
  resolved to `robot0_right_hand`. Hand-to-site rotation error was
  `7.884953353001449e-08 rad`; EEF displacement was
  `0.010280473957644904 m`.
- Raw evidence:
  `/home/hjaber/EmbodimentSemantic_archive/robocasa/calibration_gpt6_1922134/probe.json`.
- This remained a separate unscored diagnosis. It did not load MolmoPoint or
  produce an official task-success result.
- Independent full local validation: 79 passed, 4 OpenCV-dependent skips.
  A final direct-API full-mode motion guard and its regression increased the
  suite to 80 passed, 4 skipped. Python compilation and launcher shell syntax
  passed. The OpenCV tests will run in the existing compute allocation.

## Scored canary (completed; task failure)

- Job `1922136`, source `d8dbc58ed3cbf87612756184a9db4fb2faa57347`,
  `PickPlaceCounterToStove`, seed 1000, one episode, official target split.
- Calibration and the pinned MolmoPoint runtime executed. Official success
  was false. The controller performed five gripper-preshape hold actions and
  stopped at `no_candidates`; these actions did not constitute meaningful arm
  motion or an attempted grasp.
- Both decoded Molmo points were rejected as outside the observed object
  mask/snap radius. All 144 deterministic fallback candidates were rejected
  for `approach_obstruction`.
- The overlay and stored geometry align the source/arrow/mask with the steak
  near pixel (71, 166); the model points lie on the robot near (174, 129).
  The image/point decoder audit found no evidence of an axis or flip error.
- Both jaw symmetries collided with observed steak/plate points during the
  approach. This is consistent with a limitation of the frozen rim-based
  grasp policy on this scene. Clearance, candidate generation, prompt
  structure, token limit, and ranking were not tuned in response.
- The generated model text reached the configured 200-token limit and was
  unfinished; the canonical decoder returned two points. This failure is
  preserved rather than treated as positive grasp-performance evidence.
- Raw result, detailed candidate rejection records, and image:
  `/home/hjaber/EmbodimentSemantic_archive/robocasa/scored_canary_gpt6_1922136/run/results/`.

## Previous execution in one allocation

- Per the user's instruction, all live evaluation stages shared one persistent
  SLURM allocation. The original diagnostic chain used `1922154`; the adapted
  matrix used `1922186`. No per-test or per-task SLURM jobs were submitted.
- Session:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/robocasa/sessions/gpt6_unified_20260906`.
- Durable session archive:
  `/home/hjaber/EmbodimentSemantic_archive/robocasa_sessions/gpt6_unified_20260906`.
- Runtime inventory passed on `compute-6-11`: Python 3.11.16, Torch 2.7.1,
  torchvision 0.22.1, NumPy 2.2.5, MuJoCo 3.3.1, robosuite 1.5.2,
  Transformers 4.57.1, LeRobot 0.3.3, and the cached pinned Molmo checkpoint.
- Review corrections preserve official outcomes on environment-close errors,
  fingerprint all production RoboCasa Python sources for resume identity,
  enforce the full 21-task / one-episode / seed-1000 protocol, validate the
  Torch stack, and reuse verified assets across source-only changes.
- The evaluation now owns one lazily loaded Molmo runtime across sequential
  cells; each cell creates a fresh perception worker and task prompt state.
- Reviewed immutable full-run source:
  `da768978adc38d9ca0528154123a56378183f4f3`. A private snapshot commit/ref
  preserved the user's checkout, index, and unrelated LIBERO changes.
- Cluster validation: 83 passed, 1 expected skip (the renderer-unavailable
  branch skips because OpenCV is present). The combined-source motion probe
  passed: ten commands, valid B0, matching controller, 10.28 mm EEF motion,
  positive Z, and no exceptions.
- The user instructed immediate execution of the 21-task matrix and repair
  from its observed failures. The remaining unscored preflight was stopped
  with SIGTERM; stage 002's nonzero termination is intentional, not a failed
  integration gate. Stage 003 runs the full public evaluation entrypoint
  directly, without another smoke-test series.
- Full output:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/robocasa/sessions/gpt6_unified_20260906/full21/results`.
- Full durable archive:
  `/home/hjaber/EmbodimentSemantic_archive/robocasa/full21_gpt6_unified_1922154`.
- Full 21-task execution completed with 0/21 official successes: 20 task
  failures and one geometry-contract failure. Seventeen cells stopped at
  `no_candidates`; three made substantial arm motion but timed out in a motion
  phase. The remaining cell failed role projection before motion.

## Adapted object-contact run (completed; exploratory)

- Source snapshot: `e4d9683a110944cf95719423266aa5e5c37a38e8`, clean remote
  release `/home/hjaber/EmbodimentSemantic_releases/e4d9683a110944cf95719423266aa5e5c37a38e8`.
- Job `1922186`, compute-5-11, one four-hour GPU allocation, one direct stage,
  profile `object_contact_v1`, all 21 tasks, one episode each, seed 1000,
  target split, pinned MolmoPoint revision. Job and session exited cleanly.
- Result: **0/21 official successes**, all 21 terminal. The accounting was
  19 `task_failure` and 2 `geometry_contract_failure`. The live runner wrote
  every row and retained the planned denominator.
- Of the 19 task failures, all stopped at five preshape actions with no
  executable candidate. Six cells admitted at least one Molmo point through
  the metric arrow gate; those points produced 90 candidate-grid attempts in
  total, all rejected by `approach_obstruction`. The other 13 task failures
  had every decoded point outside the 23.5 mm arrow-anchor gate, so no local
  geometry was generated. The two geometry failures were role projection:
  `obj` in `PickPlaceCounterToCabinet` and `container` in
  `PickPlaceSinkToCounter`.
- In `MakeIcedCoffee`, the adapted prompt named `ice cube`, two points were
  admitted near the source anchor, and all 18 generated candidates were
  rejected by the measured compiled-finger approach obstruction. This fixes
  the earlier diagnostic's bowl-rim mislocalization at the proposal stage but
  does not yield a physically feasible grasp.
- The adapted run did not reach arm motion or official task evaluation in any
  cell. It therefore provides no evidence of improved task success, retention,
  or transfer. It shows a change in proposal behavior and exposes a remaining
  approach-geometry bottleneck.
- Durable stage archive:
  `/home/hjaber/EmbodimentSemantic_archive/robocasa/object_contact_full21_retry_gpt6_resume/run/`.
  Durable session archive:
  `/home/hjaber/EmbodimentSemantic_archive/robocasa_sessions/gpt6_object_contact_resume_20260906/`.
  The `results.jsonl` SHA256 is
  `8e821331a8ab5c71cb50719751a550651c0e06914e92b108d4a2c060878e9a6f`.
- Content-addressed input images, prompts, Molmo points, local-mask hashes,
  complete candidate rejection details, motion diagnostics, and projection
  failure frames are preserved per cell. The session allocation was released
  after archive completion at `2026-09-06T21:35:38Z`.

## Historical canary diagnosis and completed exploratory treatment

- In MakeIcedCoffee, the actual source is an ice cube inside a bowl. The
  noun-adapted canonical prompt still requested the cube's rim, and the broad
  7.5 cm RGB-D region included the surrounding bowl. The selected contact was
  outside the cube's projected box. This is evidence of inappropriate contact
  selection for this source, not an image-axis or point-decoder error.
- A separate motion-frame bug was identified: the mobile base can drift during
  arm motion despite zero commanded base channels. Raw EEF observations in the
  current base frame were previously labeled as the frozen reset frame B0.
  The adapter now composes observations into B0 and rotates B0 actions into the
  current OSC base frame, with raw sensors and sent commands retained.
- The user explicitly authorized prompt and object-grasp adaptation while
  preserving the LIBERO motion pipeline and zero training. `object_contact_v1`
  queries Molmo on the same-frame green-arrow RGB image, requests visible
  object grasp contacts, admits points by metric proximity to the source
  endpoint, and constructs local observed RGB-D support for the existing
  candidate engine. It adds no simulator masks, object poses, or task-specific
  grasp handlers to the policy.
- Canonical configuration, policy lock, candidate geometry, collision checks,
  ranking contract, motion phases, and recovery remain available unchanged.
  The adapted profile has a separate identity and a new output directory.
  Resume now rejects different experiment identities instead of discarding
  their existing rows.
- The completed 21-task pass combined the frame correction and perception
  treatment. It is exploratory development evidence; comparison with the prior
  broken-frame run cannot isolate a grasp-policy effect. A causal transfer study
  would need paired frame-correct canonical and adapted runs on untouched
  seeds/tasks.
- LIBERO records support the bundled bowl-transfer policy at 87/100 and
  429/500 successes. They do not establish the contribution of any single
  component or guarantee generalization to RoboCasa objects. Broad VLM-point
  plus geometry decomposition is established prior art; novelty remains an
  open research question being investigated separately.

## Current checkpoint

- The user asked to proceed after the pause; the planned adapted run has now
  completed and no live work remains scheduled.
- `1922154` is **COMPLETED, exit 0:0**, confirmed with `scontrol`; it is absent
  from `squeue`. Accounting queries through `sacct` were unavailable, so the
  scheduler controller and session completion files supplied terminal evidence.
- Scratch and durable session archives both contain `SESSION_DONE` with
  `finished_utc=2026-09-06T21:06:24Z` and `job_id=1922154`.
- Stage 003, the original-policy full run, completed in `1922154`. The adapted
  retry stage completed in `1922186`; its first empty-output attempt is retained
  as a failed pre-experiment gate. All 21 adapted live episodes executed.
- The reviewed code transfer finished during shutdown. Its clean remote
  release is `/home/hjaber/EmbodimentSemantic_releases/e4d9683a110944cf95719423266aa5e5c37a38e8`.
- Local private ref: `refs/robocasa/gpt6-object-contact-20260906`.
  Bundle: `.codex/legion-local/jobs/robocasa_object_contact_20260906.bundle`.
  The private snapshot preserved the user's HEAD, normal index, and unrelated
  LIBERO edits. It is not the checked-out branch.
- Latest validation: independent suite 16 passed; independent review reran
  object-contact and independent tests, 25 passed. The full 107-test RoboCasa
  suite yielded 103 passed/4 OpenCV-dependent skips in the tester environment
  and 106 passed/1 expected skip with OpenCV available. Compilation and diff
  checks passed. Reviewer verdict: APPROVE_WITH_NOTES, no remaining blocking,
  high, or medium findings in the reviewed adaptation.
- Real candidate-generation fixtures cover a compact 20 mm by 8 mm raised
  solid and a rim-like patch: finite on-target contact, short-axis jaw
  alignment, aperture clearance, and obstruction audit. These fixtures use a
  parked hand sphere; they do not substitute for the full live Panda envelope.
- All assigned research, implementation, testing, and review agents completed.
  There is no scheduled continuation. The next action is analysis/design of a
  controlled follow-up, not another unpaired 21-task run.

## Next research and repair plan

### 1. Re-establish the exact checkpoint; do not rediscover the project

Read this record and the original handoff. Inspect local Git state and the
private ref above; preserve unrelated uncommitted files. Use the existing
Legion wrappers through mp4. Verify the remote release SHA and cleanliness,
the pinned environment, cached Molmo revision, available allocation, and output
locations. Do not rebuild dependencies or redownload the model unless a
specific check proves they are missing or incompatible.

The existing reviewed runtime is:

- Python environment: `/home/hjaber/.conda/envs/robocasa-pickplace-py311`.
- Molmo: `allenai/MolmoPoint-8B`, revision
  `188130f961c8e0888a34e11121a1423c461a01ba`.
- Transformers cache:
  `/home/hjaber/EmbodimentSemantic_runtime/EmbodimentSemantic/v9d_molmo/cache/transformers`.
- RoboCasa revision `4f8a2980def75a55dff96b990745b83540425f09`;
  robosuite revision `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.
- Torch 2.7.1, torchvision 0.22.1, NumPy 2.2.5, MuJoCo 3.3.1,
  Transformers 4.57.1. Full runtime inventory remains in the session archive.

Completion condition: the intended source, runtime, model, profile, and output
root are identified without modifying an active or dirty shared checkout.

### 2. Design the controlled follow-up before another live run

The first adapted full matrix is complete and must be treated as development
evidence. Before spending more GPU time, define a paired comparison that can
separate the frame correction from the contact-proposal change. Use a fresh
source/profile/output identity and, if live execution is needed, one new
four-hour allocation containing the complete comparison stages. Submit no
separate smoke, probe, or per-test jobs.

The adapted command that produced the completed result was:

```bash
python -m vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint \
  --mode full --execute-motion --grasp-profile object_contact_v1 \
  --tasks all --episodes-per-task 1 --seed-base 1000 --split target \
  --output-dir "$RUN_ROOT/results"
```

Use the pinned Python executable through the session driver. Load Molmo once
per matrix and reuse it across fresh episode workers. Keep frame-corrected
`canonical_rim` and `object_contact_v1` paired on the same task/seed cells;
preserve all terminal rows, diagnostics, and distinct output roots. The adapted
run took about 16 minutes after allocation start, including the one-time model
load. Do not reuse its output directory.

Completion condition: the comparison protocol and changed factor are fixed
before launch, with enough untouched seeds/tasks to support the intended claim.

### 3. Use the completed matrix to design the next repair

The completed matrix already supplies the first failure-stage table. MakeIcedCoffee
is the representative cube-in-bowl case: its adapted prompt and arrow-tail points
are on the intended source, but all 18 candidates fail the measured compiled-finger
approach-obstruction check. The next repair should inspect that approach geometry
across representative solid and rim tasks, plus the two projection failures, before
changing the candidate engine. The next profile is a bounded approach-hypothesis
treatment: preserve world-down and add a calibrated camera-ray approach candidate,
then run both through the unchanged aperture, collision, ranking, and motion
contracts. For each candidate failure, answer in order:

1. Is the source correctly identified in the actual Molmo input, and is the
   decoded point on that source? Use stored images and pixels. Simulator
   annotations may be used only for clearly labeled offline diagnostic truth,
   never as new policy inputs.
2. Does the point have valid aligned depth and pass the metric arrow gate?
   Does its local patch include the source or a neighboring bowl/support?
3. Are candidates generated? If rejected, inspect the exact blocking point,
   hand primitive, aperture, workspace, and approach sweep. Preserve physical
   collision checks; do not silently loosen them to improve success counts.
4. Does the frame-corrected arm reach its pregrasp and contact targets? Compare
   B0 observations, raw base drift, current-base commands, and compiled-site
   diagnostics. Separate coordinate defects from reachable-space/controller
   limits and physical collisions.
5. Does close-and-lift retain the intended object, then place, release, retreat,
   and satisfy the official task predicate? A selected candidate, arm movement,
   or a successful lift alone is not task success.

Create a compact failure table and show representative saved frames for the
cube/bowl case. If a concrete implementation defect is found, repair the
responsible boundary, run only relevant inexpensive local validation, obtain
independent review, and freeze a new source/output identity. Any rerun belongs
inside the same active allocation. Do not create an endless sequence of
individual live tests or overwrite earlier evidence.

### 4. Preserve the LIBERO mechanism while testing its actual limits

Keep the frozen Molmo checkpoint, RGB-D grounding, deterministic candidate
ranking, measured robot geometry, aperture/collision checks, motion phases,
action scaling, recovery, and evaluation timing. The current adaptation changes
the noun-conditioned contact prompt and local support selection. It adds no
training or task/category-specific grasp code.

The mathematical carryover is local shape direction to jaw orientation and
feasibility: a small solid's major axis gives a short-axis straddle; a local
rim arc gives a tangent/radial jaw relationship. A legacy `rim` variable name
does not alone prove failure on solids. However, the fixed vertical approach
and 15 mm support radius are real restrictions. Wider objects can have truncated
width estimates, and thin objects can merge with nearby support in the upper
depth band. If the full matrix demonstrates these are the remaining bottleneck,
the next design should generalize contact geometry below the proposal boundary
with an explicit new profile, rather than add named-object exceptions.

Do not modify or launch LIBERO during the immediate RoboCasa repair. A later
cross-suite experiment can port the same profile deliberately after its behavior
and scientific question are clear.

### 5. Establish a controlled transfer experiment before making research claims

The existing 0/21 run is a preserved diagnostic result with a frame bug. The
completed treatment combines the frame correction and contact selection change;
its result cannot isolate either effect. Seed 1000 scenes have been inspected
and used for development, so they cannot be relabeled as untouched evaluation.

Once the pipeline is usable, specify the hypothesis, changed factor, metrics,
and untouched evaluation cells in advance. The minimum useful comparison is
frame-corrected `canonical_rim` versus `object_contact_v1`, with paired resets,
seeds, runtime, model, action budgets, recovery, and official success criteria.
Include bowl controls and held-out solid-object cases. Then test the same
adaptation in LIBERO without suite-specific tuning. The current full-mode
runner intentionally locks the 21-by-1 seed-1000 development contract; a larger
held-out study requires an explicitly designed protocol and identity, not an
unrecorded edit to that guard.

Report on-object proposal rate, feasible-candidate rate, grasp retention,
official success, failure category, sample count, and uncertainty. Improved
candidate availability without better execution identifies an unresolved
geometry/control bottleneck. A high score from one inspected seed per task is
not sufficient evidence for broad generalization or publication readiness.

### 6. Use the paper reviews to narrow a falsifiable research direction

Both requested papers were read by separate research agents:

- [MOKA, RSS 2024](https://www.roboticsproceedings.org/rss20/p062.pdf): selects
  role-labeled semantic points, grounds them with depth, and connects them to
  analytical antipodal grasp/motion generation. Its zero-shot path requires no
  new task training, but its implementation uses Grounded-SAM and a different
  execution stack. The reusable lesson is to evaluate semantic point selection
  separately from physical grasp feasibility. Its broad decomposition is close
  prior art, not a novel claim available to this project.
- [Robotic Visual Instruction (RoVI), CVPR 2025](https://cvpr.thecvf.com/virtual/2025/poster/34129):
  uses visual symbols/keypoints, RGB-D grounding, VLM-generated skills, and
  grasp execution. Its full VIEW stack adds a trained YOLO keypoint detector
  and AnyGrasp. Our paired clean/arrow images already permit deterministic
  symbol extraction, so importing that detector would add complexity without
  fixing the demonstrated grasp issue. Its arrow-to-execution concept also
  overlaps prior art. [Project and release information](https://robotic-visual-instruction.github.io/).

The wider side-agent search proposed three future hypotheses: source-specific
post-lift verification, normalized local contact representations, and an
uncertainty-triggered second observation. None has established novelty or a
verified benefit in either suite. Do not add them to the pending treatment.

The most useful diagnostic follow-up is whether visible evidence can distinguish
the intended object from an accidentally grasped container. A simple co-motion
test is insufficient: lifting a bowl can carry the cube with it. A viable
experiment must include nested-object counterexamples and explicit ambiguity
or abstention, measuring wrong-object acceptance at matched coverage. Active
perception and object-relative representations also have substantial prior art;
they should remain candidate ablations unless stronger evidence justifies them.

Completion condition for the research-design phase: one precise cross-suite
hypothesis, its nearest prior work, minimal implementation delta, controls,
falsifier, and held-out protocol. Do not equate integration work or literature
overlap with an ICLR contribution.

### 7. Deliver evidence and close resources

On any next completed run, report the actual success count and denominator,
remaining failure categories, corrected versus unresolved mechanisms, source
and profile identity, and links to the cube/bowl images and archived results.
Update this same record. Release the session allocation after its intended
stages; do not leave an idle GPU running or create an automatic continuation.

Current completion is deliberately separated: original matrix 21/21 terminal;
adaptation implementation, independent review, and adapted live matrix complete;
both live matrices have 0/21 official successes; requested paper reviews 2/2
complete; generalization and novelty claims remain unproven. No honest single
percentage describes all of those.
