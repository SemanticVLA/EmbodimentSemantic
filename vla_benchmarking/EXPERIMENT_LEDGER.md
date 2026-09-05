# SmolVLA Arrow Experiments: Authoritative Takeover Ledger

## Canonical cleanup regression run — 2026-09-05 17:48:06 UTC

The organized canonical controller is being re-evaluated on **500**
sealed-randomized cells: tasks 0-9, 50 episodes per task, and deterministic
seeds 1000-1049 for every task. Full Legion job **1921206** is RUNNING on one
A40 from immutable commit
`d044a5d672092d4322cf7395d91cc1b0085ef496`. Its label is
`canonical_grasp_sealed500_d044a5d_20260905T1748Z`; archive target:
`/home/hjaber/EmbodimentSemantic_archive/grasp_controller/canonical_grasp_sealed500_d044a5d_20260905T1748Z_1921206`.

Dependency smoke job **1921205** completed **1/1** successfully and archived
with status `VERIFIED`. It exercised the pinned MolmoPoint model, aligned
agentview RGB-D path, candidate generation, 40 mm preshape, placement, opening,
80 mm retreat, and post-retreat evaluator. The exact canonical configuration
hash remained
`37497fd0b2f60346b9ffd1501ccc743046c7fa2370ef6fa9531a7204f69cc044`.

Initial job **1921204** is a preserved pre-workload failure: the reorganized
launcher's early policy-hash check could not import the organized package. It
ran no episode and produced no evaluation result. Commit `d044a5d` adds only
the missing repository-root import bootstrap plus its regression assertion;
controller behavior and configuration are unchanged. The 500-cell result is
a refactor regression check, not a new treatment or automatic replacement of
the historical 87/100 evidence.

## Canonical grasp controller — 2026-09-05 17:27:12 UTC audit

The sole active grasp-controller treatment is `failure_opening40_retreat80`,
verified on the 100-cell sealed-randomized matrix as **87/100 (87%)** in Legion
job **1920556**. Per-task successes were T0 10/10, T1 10/10, T2 10/10,
T3 10/10, T4 7/10, T5 10/10, T6 8/10, T7 10/10, T8 10/10, and T9 2/10.
This terminal record supersedes older ledger snapshots that call the 20/60 e8
canary the winner; those entries remain below as time-stamped history only.

The executed source release was
`b4fb87759ae3a1ea2cd518cd201a1a737bb14e80`. Scientific identity
`717d00640ab5b73fed0dcbbf9d7703786d2a527efc2e02205d16f34bbc13465a`;
underlying controller digest
`60f4f5f9ecfde7b4830f376ab06cfc706e2ef175d86817c42a0adb7cddd46c0c`;
canonical configuration SHA-256
`37497fd0b2f60346b9ffd1501ccc743046c7fa2370ef6fa9531a7204f69cc044`.
Summary, manifest, status, and runtime-version SHA-256 values are respectively
`569ac1e8400af9f80047fd8f6a6060946c3c0e80dfaeb348e04b655e51d3472a`,
`eb0788ce41e10215f4ee730d53d52360cd1731a6e6a91999af385babd127b369`,
`b15e0bd6b9bda542a7c7662723a22ac36f9181ae2e2e3b0320a5661370a7cd52`, and
`dcb7f35401b81cb634b23ced3866b7124c8bab54b3f79bdecfd2372a4fcf1ec3`.
The immutable archive is
`/home/hjaber/EmbodimentSemantic_archive/molmo_failure_sealed100/molmo_failure_sealed100_fa1ae83_1920556`.
The 87% result belongs to the executed release; the organized default is
behavior-equivalent code and is not claimed as a separately rerun experiment.

## Placement XY terminal correction — 2026-09-05 17:27:12 UTC audit

Legion job **1920138** completed its 12-cell screen with exit code 0:
**3/12**, 12 terminal, retained-lift heuristic 5, successful retries 0,
2,866 reported actions, and 1,222.745 seconds of perception latency. Vanilla
was 2/6 and sealed-randomized was 1/6. Failures were no_candidates 4,
recovery_failed 3, task_failure 1, and input_failure 1. It did not outperform
the matching 5/12 reference prefix and was not extended. Its archive remains
`COPIED_UNVERIFIED` at
`/home/hjaber/EmbodimentSemantic_archive/v9d_molmo_visualxy12/v9d_molmo_parked_visualxy_e8fcc99_1920138`.
This terminal correction supersedes the earlier RUNNING snapshots immediately
below without deleting their historical observations.

## Placement XY partial — 2026-09-04 10:30:24 UTC

Job1920138 RUNNING on compute-3-14, elapsed16:01. **3 successes /12 planned,
10 terminal**, two remaining (sealed T9/1000 running, T9/1001 planned).
Vanilla2/6, sealed1/6 planned. Retained-lift heuristic4, successful retries0,
1483 reported actions, no unknown action counts,686.648s perception.
Failure categories:no_candidates4,task_failure1,recovery_failed1,input_failure1.
Sealed T6/1000 failed with ambiguous arrow direction; preserve it, no new glyph
tuning. No arm stop recorded yet. Maximum possible final successes5/12 only
ties the winner's matching screen; complete remaining cells and apply the
predefined tie-breakers if needed. No improvement claim or full60 extension yet.
Winner remains20/60 unchanged. CPU archive verification1920120 still pending
Priority; no duplicate verification or GPU submission.

## Placement XY live startup — 2026-09-04 10:17:14 UTC

Job1920138 is RUNNING on compute-3-14 (elapsed2:50). Verified actual allocation
reports one NVIDIA A40, BF16 support, and exact pinned Torch/Transformers and
other package versions. All8 Molmo checkpoint shards loaded. Result counts
not inspected in this startup check; do not infer canary improvement yet.
CPU archive verification1920120 remains pending Priority. Ten-minute heartbeat
updated and ACTIVE; winning Molmo-dense20/60 is preserved, not replaced.

## Placement XY ablation submitted — 2026-09-04 10:13:53 UTC

Job **1920138 PENDING (Priority)**: one A40, eight CPUs, 64 GB, four-hour cap;
exactly 12 planned cells (T4/T6/T9, seeds1000/1001, vanilla/sealed_randomized).
Fresh label `v9d_molmo_parked_visualxy_e8fcc99`, same clean immutable e8fcc99
source, dense_agentview_clearance arm, rim_clearance prompt, parked observation,
full_open, release20_visual_xy, RGB-D/no SAM, explicit screen-only.
No default changes and no automatic full60 extension. Compare against the
winning full-open arm's matching 5/12 (retained heuristic6,2788actions,949.150s
perception); full60 winner remains20/60. Advance only if eligible and stronger
under successes/planned then retained lifts, actions, latency; otherwise pivot.

Operator `.codex/legion-local/molmo-parked-visualxy-e8fcc99/launch_visualxy_12.sbatch`
SHA256 `59ef86da239fd52c8689307121717aef2ae3119b2acab8207ece1634ace4a8bd`,
matched exactly on Legion before submission; remote immutable operator directory
`/home/hjaber/EmbodimentSemantic_runtime/operator/molmo-parked-visualxy-e8fcc99-59ef86da`.
Submission lock/receipt prevents duplicates. Targeted22 tests and independent
reviewer APPROVE/bash syntax passed; no production-code change or full-suite loop.

Run root: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/visualxy12/v9d_molmo_parked_visualxy_e8fcc99_1920138_run`.
Output root: same parent, `v9d_molmo_parked_visualxy_e8fcc99_1920138_results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo_visualxy12/v9d_molmo_parked_visualxy_e8fcc99_1920138`.
Logs: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/molmo_visualxy12_1920138.out` and `.err`.
Archive is COPIED_UNVERIFIED until independently checked after completion.
Runtime/startup/results are not yet verified: scheduler submission is not task success.

## Continued improvement authorized — 2026-09-04 10:05 UTC

User explicitly requests continued ten-minute repair/test work beyond the first
improvement. Keep the existing heartbeat ACTIVE; 20/60 is now the score to beat.
Preserve both the winning e8fcc99 release and frozen fd24 v9d. No timer-only
code changes, running-release mutations, default promotion, or 200-cell expansion.

Next: a fresh 12-cell parked/full-open dense-agentview rim_clearance run using
existing release20_visual_xy instead of release_plus20mm. This single-factor
placement-anchor ablation shifts placement XY by (+20.3,-5.2) mm while preserving
grasping, opening, yaw, Z, release height, camera, model, and action budgets.
Compare with the winner's matching 5/12 prefix, not its aggregate 20/60.
Architect recommended the existing profile: no production changes needed.
Preparation in progress; no new GPU job submitted yet. CPU archive verification
1920120 remains pending (Priority) at 10:05:24 UTC.

Post-hoc audit: 12 completed placement/retreat sequences ended task_failure
(T4:2, T6:3, T9:7). Another overlapping 12 cells had a retained-lift heuristic
followed by a placement-phase timeout (T4:0, T6:5, T9:7). These observations
justify a placement experiment, not a claim that the offset is wrong or every
retention heuristic represented a physically held bowl. No evaluator/object
state is used to select online targets.

Root inspected observed retreat images for sealed T4/1007, T6/1009 and T9/1004
(local ignored placement-evidence-1919325 directory). They do not establish a
systematic XY bias; gripper occlusion limits interpretation. T9/1004 open EEF
was within 0.751 mm of its requested release target. Keep this an exploratory
profile comparison, not a verified geometry repair. Independent targeted
transfer-XY/integration/motion-profile/identity tests: 22 passed.

## Completed canary improvement — 2026-09-04 09:52:51 UTC audit

Job1919325 finished at05:01:33UTC, campaignstatuspartial_failed/exit2 because
the40mm arm reached its repeated-operational-failure stop. The full-open arm
continued and completed ALL60planned cells with returncode0,no stop/fatal:
**20/60 =33.33%**, versus historical15/60=25% and14/60=23.33%. That is5more
successes/+8.33percentage points over the better historical baseline. This is
an observed exploratory canary improvement across different code revisions,
not isolated proof of Molmo's causal effect or generalization.

Winner: dense agentview MolmoPoint, rim_clearance prompt, parked observation,
full_open, release_plus20mm, adaptive_short_v1, no SAM. Immutable source
e8fcc99aacf962f8f43e14b25111f9df46c0c944, scientificidentity
8db821deba40ff43a565b7a94a4893729baa08671017004bdacd26a2573ef176.
Vanilla11/30, sealed9/30. T4/T6/T9 totals4/20,15/20,1/20 respectively.
Retained-lift heuristic40, successfulretries3,17238actions,5480.213s perception.
Failures:no_candidates13,task_failure12,grasp_failed10,recovery_failed5.
All reported per-cell actions<=1200 (vanilla max933,sealed max1000),noneunknown.
The15new successes supplement5preserved prefix successes;48new cells executed.

40mm stopped with **16/60 planned,25terminal,35unattempted** after vanilla
T6/1006 repeated the carried sealedT6/1000 ambiguous-arrow error.13newcells,
retainedheuristic17,retry successes3,7108actions,1494.742s perception;
vanilla13/30planned,sealed3/30planned. Failures:no_candidates3,grasp_failed1,
input_failure2,task_failure1,recovery_failed2. No more renderer tuning or
replacement of its stopped results. It is not a completed60-cell result.

No active GPU runs at audit. Durable archive exists asCOPIED_UNVERIFIED;
CPU-only verificationjob1920120 submitted09:55:04 to compare allrun/output
file hashes and exact preservedprefixrecords. No new GPU experiments or
defaultpromotion. Frozenfd24a4c v9d and main-checkout user edits untouched.

## Full60 continuation partial — 2026-09-04 03:25:34 UTC

Job **1919325 RUNNING** on compute-3-12.40mm has **15/60 planned successes,
23terminal cells**, including the12preserved prefix cells plus11new cells.
The new cells contributed9successes. Retained-lift heuristic16,successful
retries2,5999reported actions,1326.706s perception; failure categories
no_candidates3,grasp_failed1,task_failure1,recovery_failed2,input_failure1.
Per-suite successes: vanilla12/30planned,sealed3/30planned. Only vanilla
extension cells have executed so far; full-open extension has not started.
No new operational-stop evidence; original input failure remains counted.

This partial count matches the better historical baseline's final15/60 total,
but37cells remain and historical commits differ. Do not declare a completed
canary improvement or promote the default. Continue the current immutable run
unchanged; no repair or new launch is justified by this audit.

## Full60 continuation live verified — 2026-09-04 03:13:07 UTC

Job **1919325 RUNNING** on compute-3-12; submitted03:10:25,started03:11:23.
One NVIDIA A40/BF16 and all pinned packages verified, including Torch2.10.0
and Transformers4.57.1; Molmo checkpoint shards loaded. Continuation manifest
records exacte8fcc99 execution and matching40mm scientific identity. The first
NEW vanillaT4/1002 cell is RUNNING; all prior prefix cells are present unchanged,
including sealedT6/1000 failed. No new terminal result at this audit. This is
verified live resume, not merely a submitted job; full-open follows sequentially.

## Continuation path correction resubmitted — 2026-09-04 03:10:24 UTC

Job **1919325** submitted with the SAME reviewed operator/source and disjoint
run/results directories. Job1919324 failed before any workload or trial at
03:08:23(exit2,elapsed0): root supplied OUTPUT_ROOT inside RUN_ROOT, which the
wrapper correctly rejected. This was an operator submission mistake, not a
canary outcome. Its logs are preserved. No prior trial was executed or replaced.

Active label `v9d_molmo_full60_e8fcc99_op8d4af55c_r2`; do not duplicate.
Run: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/full60/v9d_molmo_full60_e8fcc99_op8d4af55c_r2_run`.
Results: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/full60/v9d_molmo_full60_e8fcc99_op8d4af55c_r2_results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo_full60/v9d_molmo_full60_e8fcc99_op8d4af55c_r2_1919325`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/molmo_parked_full60_1919325.out`.
All prefix preservation,48newcells/arm,shared stop history and4h/A40 limits
remain unchanged. This job has no new canary result at submission.

## Same-SHA full60 continuation submitted — 2026-09-04 03:07:33 UTC

Job **1919324** submitted successfully and is PENDING(Priority) at this audit.
One A40/eight CPUs/64GB/four hours, sequential40mm then full-open, one persistent
Molmo worker. Both eligible treatments advance to60planned cells each; retain
their12terminal prefix cells and run only48new cells per arm (96new total).
No SAM, no scientific-code revision, no default promotion, no200-cell expansion.

Prefix job1919323 completed with exit0 and VERIFIED durable archive (03:06:30).
Final40mm **6/12**, vanilla3/6,sealed3/6,retained evidence7,retry successes1,
3231actions,683.125s perception. Full-open **5/12**, vanilla2/6,sealed3/6,
retained evidence6,retry successes2,2788actions,949.150s perception;
failuresno_candidates4,recovery_failed2,task_failure1. Both arms12terminal,
returncode0,no stopreason/fatal. Historicalmatchingprefixes3/12each are from
different commits; these are screen improvements, not final60-cell claims.

External operator independently APPROVED; six focused tests include REAL matrix
copy/resume48new calls, prefixfailure/interruption preservation, sourcebyte
immutability and realStopRules history/deduplication. Existing identity/campaign
checks50passed. Live `_validate_prefix` accepted BOTH complete Legionarms before
submission. Prefix errors are replayed so their stop history is not reset.
The original prefix remains untouched; continuation copies into a NEW root.

Execution remains clean `e8fcc99aacf962f8f43e14b25111f9df46c0c944`.
Operator driverSHA256 `8d4af55ccc27f79fa4b346006ea15c3d717656d79e18c5c0318bfa600169a451`;
sbatchSHA256 `c9264d539d574b7eace6d7e93545e0fbb76c3a44024a487f221d7831abbd2062`.
These are separately archived operator artifacts, not changes inside thatcommit.
Operator directory: `/home/hjaber/EmbodimentSemantic_runtime/operator/molmo-full60-e8fcc99-8d4af55c`.
Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/full60/v9d_molmo_full60_e8fcc99_op8d4af55c/results`.
Archive target: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo_full60/v9d_molmo_full60_e8fcc99_op8d4af55c_1919324`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/molmo_parked_full60_1919324.out`.
Do not resubmit. This wrapper marks archives COPIED_UNVERIFIED until independent
post-run content-hash verification. SLURM accounting service was unavailable
at03:06:30; prefix exit/archive evidence and fresh scheduler state were available.

## Short-arrow 40 mm screen complete — 2026-09-04 02:50:03 UTC

Job1919323 remains RUNNING on compute-3-12; the full-open arm is now running.
The40mm arm completed all12planned cells with returncode0,no stop reason,
and no fatal flag: **6/12 successes**, vanilla3/6,sealed3/6,retained-lift
evidence7,one successful retry,3231actions,683.125s perception. Failures:
input_failure1,no_candidates2,recovery_failed2,task_failure1. Retention is a
proprioceptive heuristic, not physical proof. Full-open has4terminal vanilla
cells withT9/1000running; no final comparison yet.

Live short-arrow evidence: sealedT9/1000 refresh03/04 decoded the14.765px
arrow and the retry placed successfully; sealedT6/1000 refresh03 still failed
closed on the10.198px ambiguous arrow. That failed terminal cell stays counted.
One targeted renderer repair was used; a second matching operational failure
in the same extended cohort triggers the stop rule, not more glyph tuning.

The complete40mm screen qualifies for a separately recorded same-SHA full60
continuation; preparing an external no-SAM operator driver without changing
the immutablee8fcc99 runner. Copy to a NEW root, preserve all12terminal cells,
replay their StopRules history, and execute only48new cells. No submission yet.
The historical matching prefixes were3/12 each, but their commits differ;
6/12 is promising exploratory screen evidence, not a completed60-cell win.

## Short-arrow canary first live result — 2026-09-04 02:33:14 UTC

Job1919323 remains RUNNING.40mm arm:1/12planned,2finished; vanillaT4/1000
placed successfully(222actions),T4/1001no evaluator result(64actions),T6/1000
running. New clean/rendered frame hashes and decode audits are present on
Legion; observed first arrows use the unchanged long-arrow branch. The short
retry branch has not yet been observed in this audit. Full-open arm not started.

## Short-arrow paired canary launched — 2026-09-04 02:30:40 UTC

Job **1919323** is RUNNING on compute-3-12, submitted02:30:16 and started
02:30:22 UTC. One A40/eight CPUs/64GB,two-hour limit. Clean immutable source
`e8fcc99aacf962f8f43e14b25111f9df46c0c944`, label
`v9d_molmo_shortarrow_e8fcc99`, matching source/runtime,CUDA,BF16,Torch2.10.0
and Transformers4.57.1 verified. First parked_opening40mm arm started.
No task results at startup audit. Both12-cell arms use adaptive_short_v1;
no SAM, no automatic extension, no replacement of prior failed results.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_shortarrow_e8fcc99_1919323/results`.
Archive target: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_shortarrow_e8fcc99_1919323`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919323.out`.
Bundle SHA256: `61efad938746768440d0c1b4499a9f896bffc17cceebc92e28cff599c217322b`.
Do not resubmit. One targeted short-arrow repair has now been used; if the same
failure persists twice, stop the arm and pivot from evidence without decoder
relaxation or per-cell arrow-style tuning. Prior5/12 is promising, not a full60 win.

## Parked paired screen terminal — 2026-09-04 02:16:30 UTC

Job1919322 is terminal, exit2, durable archive VERIFIED. Both arms stopped
after11attempted cells on the same repeated retry arrow-decode failure.
Final40mm5/12planned versus full-open3/12planned. Full-open: vanilla2/6,
sealed1/6,retained-lift evidence4,retry successes0,1783actions,453.688s
perception; failuresno_candidates4,input_failure2,recovery_failed1,task_failure1.
No active experimental allocations at this audit. Keep failed/unattempted cells
and do not extend or restamp this stopped cohort. The input-only rendering
repair is implemented and undergoing independent targeted validation.

## Short-arrow input repair preparation — 2026-09-04 02:14 UTC

Release gate APPROVED for the prefix via the existing no-SAM launcher:
independent tester60passed/27OpenCVskipped in lightPython and29passed in the
actual OpenCV environment; reviewer independently passed168actualrender cases
and found no HIGH/BLOCKER. Decode-failure artifact handling was inspected,
not independently executed. No further review round before this bounded run.

Root validation:45 focused tests passed in the installed vla_bench_py312 conda
environment with real OpenCV4.13 (pytest plugin autoload disabled to avoid an
unrelated Hydra plugin error). A separate actual-renderer/decoder sweep passed
168 span12-31px/orientation cases,max endpoint error0.5px. The10.2pxT6 arrow
still fails closed on some backgrounds (including uniformgray100); both exact
pairs pass the testedgray90 background. Do not claim universal short-arrow
repair or Legion-version equivalence before runtime evidence.

An optional future60-cell launcher-forwarding patch was excluded and reverted:
its existing setup still required SAM even for RGB-D. The upcoming paired
prefix uses the unchanged proven no-SAM campaign launcher. No extension
readiness is claimed; eligible same-SHA full60 launch needs a scoped no-SAM
job setup, not changes to scientific code or restamping old results.

Reproduced both parked retry failures using the existing renderer and unchanged
decoder with rounded source/goal centers(204,68)->(194,66) and
(193,85)->(200,72). Both legacy2px shaft/16px head arrows fail; adaptive1px
shaft and4/5px heads decode with less than0.5px endpoint error. Both prior
attempts had lifted, reached preplace160-step timeout(residual32.757/16.667mm),
then opened and retreated. Thus retries genuinely need short source-to-goal
arrows; do not infer task success from privileged relation data.

Approved input-only rule for both parked arms: rounded endpoint span<32px
uses width1 and head=max(3,min(16,round(.35*span))); longer arrows preserve
width2/head16 exactly. No endpoint movement, direct bbox targets, decoder
relaxation, geometry/motion change or evaluator shortcut. Architect reproduced
24/24 orientations at each tested span12,15,16,20,24,28,31px; tiny ambiguous
arrows still fail closed. Add explicit scientific identity and fresh frame/render
audits including decode failures. Test and review only this delta, then rerun
the paired12-cell screens at a new immutable identity; do not resume1919322.

## Parked 40 mm screen stopped with five placements — 2026-09-04 02:04:57 UTC

Job1919322 continues with the full-open arm. The40mm arm is terminal:
**5/12 planned,11 attempted**, vanilla3/6 and sealed2/6,retained-lift evidence6,
zero successful retries,2687actions,474.009s perception. Failure categories:
no_candidates2,input_failure2,recovery_failed1,task_failure1. The screen stopped
on two retry arrow-decode errors (sealedT6/1000,T9/1000), leaving sealedT9/1001
unattempted. Do not extend this incomplete stopped cohort or replace its failures.
Full-open is partial0/12,2 finished,40actions,64.851s perception,T6/1000 running.

Five placements exceeds the matching historical3/12 count, but this remains
an exploratory incomplete screen at a different commit, not a full60-cell win.
The new operational failure is after recovery, not initial perception. Current
retry bbox-center arrow spans are10.688px(T6) and15.207px(T9), versus the
fixed16px arrowhead. Preserve decoder guards and endpoint/transfer meaning;
investigate only an endpoint-preserving short-glyph rendering correction.

## Parked canary first placement — 2026-09-04 01:53:08 UTC

Job1919322 is actively evaluating the40mm arm. Two cells finished:
vanilla/T4/1000 placed successfully on the first grasp; T4/1001 produced no
candidates. T6/1000 is running. Current score **1/12 planned,2 finished**,
one retained-lift signal,zero successful retries,286actions,88.402s perception.
The full-open arm has not started. Both first cells completed44 shaping
actions at40.88649mm,total64actions before perception,without hover/settling.
This confirms the new path can place a bowl, not overall baseline improvement.

## Parked agentview paired canary launched — 2026-09-04 01:50:31 UTC

Job **1919322** is RUNNING on compute-3-12. Submitted01:50:03, started
01:50:22 UTC, one A40/eight CPUs/64GB, two-hour limit. Clean immutable release
`da9b1985a826544854b84916fe23ace91c769fe1` and BF16/Torch2.10.0/
Transformers4.57.1 runtime were verified. Frozen v9d configuration is unchanged.
Label `v9d_molmo_parked_da9b198`; no SAM. The reviewed parked protocol below
tests40mm then full-open,12 planned cells each, without automatic extension.
No completed task result at this startup audit; no improvement claimed.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_parked_da9b198_1919322/results`.
Archive target: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_parked_da9b198_1919322`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919322.out`.
Transfer bundle SHA256: `a99ee36ca6f2e11c2f24abea04e0d00065e634abec6fba1ef0d05799907374b6`.
Do not resubmit this job. Inspect actual parked shaping/candidate motion
before interpreting results; preserve stopped cells and all revision identities.

## Parked agentview pivot preparation — 2026-09-04 01:40 UTC

Release gate APPROVED: independent tester80passed5.45s, reviewer13focused
passed and no HIGH/BLOCKER. Root72profile/campaign/Bash checks plus14parked,
settled, preshape and frozen-isolation checks passed; compilation, Bashsyntax,
diff and frozenconfig comparison passed. Only the new parked delta reviewed.
Proceed directly to the bounded paired canary; live performance is unproven.

Architecture review approved omitting the entire observation-hover/settling
procedure for both matched agentview arms. Existing candidate execution
already starts from measured current EEF and stages vertical clearance,
rotation, translation, pregrasp and descent independently of observation
hover. No assertion that parked poses are universally safer or that the
previous source region was wrong is supported.

New `parked` observation profile is RGB-D agentview only and rejects settled
opening profiles. Initial full-open dwell, optional unchanged 40 mm preshape,
fresh RGB-D/current privileged arrow provider/no-motion calibration, then
ordinary candidate execution. After existing open/retreat, repeat the same
optional shaping and fresh perception. No added settling, no reuse of hover
q90 or old images; shaping drift guards and160/1200/four-grasp limits remain.

`--parked-opening-probe` tests `parked_opening40mm` then
`parked_opening_control`, 12 planned cells each, no extension. Both keep the
same MolmoPoint revision, dense agentview, rim_clearance prompt and
release_plus20mm motion profile. The comparison between these new arms
isolates opening policy under parked observation; comparison with older
hover arms is a broader observation-procedure change. Use a new immutable
identity and preserve all stopped results. Focused fake/contract tests and one
review precede the direct guarded simulator canary; no separate diagnostic.

## Settled opening screen stopped — 2026-09-04 01:33:03 UTC

Job **1919320** is terminal, `settled_opening_probe_stopped`, exit2,
archiveVERIFIED. Both arms attempted the same five vanilla cells, then stopped
on initial `opening_settling_failed` in T6/1000 and T9/1000. No sealed cells
were attempted. Preserve the incomplete 12-cell denominators; no extension.

- 40 mm: **2 successes /12 planned**, five attempted, two retained lifts,
  both successful on retry2;1299actions,229.126s perception. Successes
  T4/1000 and T6/1001; one no-candidate cell and two controller failures.
- Full-open: **1 success /12 planned**, five attempted, one retained lift,
  successful on retry2;684actions,176.695s perception. Success T6/1001;
  two no-candidate cells and two controller failures.

Narrowed opening admitted grasps and added one matched-cell placement, but
the screens are incomplete and do not establish improvement over historical
matching3/12 or full15/60 and14/60. All three successful cells first timed
out in translate_clearance, then succeeded after fresh-frame recovery.

Failure evidence: T6/1000 has normal-looking observed support q90=.945800m,
hover1.045800m, then3settling actions descend to1.026269m and21.094mm target
error. T9/1000 q90=1.176055m, hover1.276055m, then9settling actions reach
21.976mm error. Both retain just over80mm observed height margin at stop.
The20mm envelope correctly rejects both. T9 support touches image border;
robot contamination is a hypothesis, not verified object identity. Do not
generalize it to T6 or loosen guards. Next decision: a distinct observation
procedure rather than another settling threshold adjustment.

## First live canary clears repaired handoff — 2026-09-04 01:21:21 UTC

Job1919320 remains RUNNING. First treatment cell vanilla/T4/seed1000 is
running with evaluator result null. Its actual production settling completed
in15 actions (shared total63); preshape completed in44 actions (shared total107)
at40.88649mm. This reproduces the diagnostic repair inside the full canary,
without the former immediate pose_drift stop. No completed grasp/task result
yet; do not infer success from the repaired handoff alone.

## Settled opening canary RUNNING — 2026-09-04 01:19:49 UTC

Job **1919320**, submitted **01:19:18 UTC**, started **01:19:22 UTC** on
compute-3-12. One A40, eight CPUs, 64 GB, two-hour limit. Immutable clean
source **e9885480d3a7226d9b31e29df716fdc666cb1dab**, label
`v9d_molmo_settled_e988548`; remote source, CUDA device, BF16, torch 2.10.0,
Transformers 4.57.1 and first `settled_opening40mm` arm startup verified.
No completed task results at this startup audit. Do not resubmit.

Treatment then control, 12 planned cells each, screen only. Submit mode
`V9D_MOLMO_SETTLED_OPENING_PROBE=1` forces the paired campaign; raw
`V9D_MOLMO_SCREEN_ONLY=0` is expected because the paired mode itself prohibits
extension. Both use hover20mm/release_plus20mm and the same model/prompt.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_settled_e988548_1919320/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_settled_e988548_1919320`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919320.out`.
Bundle SHA256: `a270ae620eccbe4e9498650f50cfa83237345ff2da1cee4bd32b35729112cf1c`.
Scheduler wait: four seconds. Runtime package setup: one second; model loading
and perception timings remain separate. Frozen v9d and main checkout untouched.

## Settled opening comparison preparation — 2026-09-04 01:05 UTC

Final gate APPROVED: independent tester 18 passed, reviewer 20 focused checks
passed and no unresolved HIGH/BLOCKER. Reviewer found raw NumPy settling
telemetry could overwrite JSON-safe matrix evidence; shared helper now
normalizes published/returned success and failure audits. Final correction
and production integration checks: 6 passed, plus 2 actual helper-to-matrix
plain-JSON regressions with action totals preserved. Ready for one immutable
24-cell settled opening release. No additional diagnostic is required.

Implementation frozen after 104 combined targeted tests passed in 8.95 seconds,
plus compilation, Bash syntax, diff check and unchanged frozen config. Shared
helper, matrix failure evidence, launcher, campaign and legacy-profile checks
are included. Independent production retry integration and one focused final
review completed as recorded above; no further model or diagnostic setup.

Robot-only diagnostic 1919319 passed the previously stated gate. Promote its
unchanged nominal-hover settling helper into production; do not relax pose,
height, clearance, opening or action limits. New profiles `full_open_settled`
and `preshape40mm_settled` share fresh observation hover and settling on both
initial approach and retries. Only the latter performs the fixed 40 mm
preshape. Existing profiles and frozen v9d stay unchanged.

Question: does measured 40 mm opening improve the executable grasp policy
relative to full opening when both use the proven settling handoff?
Primary comparison: `settled_opening40mm` then `settled_opening_control`,
12 planned cells each (tasks 4/6/9, seeds 1000/1001, both suites). Hold pinned
MolmoPoint, dense agentview, rim_clearance prompt, hover20mm, release_plus20mm,
candidate geometry, evaluator and 160/1200/four-grasp limits constant.
Retry planning explicitly refreshes the existing privileged bbox arrow-input
provider on new RGB-D, then recomputes observed-support hover; final candidate
capture occurs after settling and optional preshape. This is not visual
tracking or a vision-only object identity claim. No simulator object poses
enter grasp geometry.

New immutable source and experiment identities are required. Preserve all
failures; stop an arm after repeated operational failure on two distinct
cells. Rank successes / planned cells, then retention, retries, actions and
latency. Screen only: no automatic 60-cell extension or default promotion.
The old full-open screens remain 2/12 versus historical matching 3/12, with
different source identities. The robot-only pass is not grasp improvement.
Remote preflight at 01:05:27 UTC found no active experimental allocation,
gpu_a40 available, HOME 249 TB free and scratch 167 TB free.

## Handoff diagnostic PASSED — 2026-09-04 00:58 UTC

Job **1919319** is terminal, exit0, archiveVERIFIED. All four cases persisted;
`control_reproduced=true`, `treatment_passed=true`. Both immediate controls
reproduce the original pose_drift failures with61/57total actions. Both
settled treatments complete unchanged preshape at **40.88649mm**, using
15/16settling actions then44preshape actions,107total actions per case.
Final settling target residual16.29/16.84mm is within original20mm tolerance;
height margins111.09/111.58mm exceed80mm, angularerrors.0649/.0670rad remain
below.12rad. Last controller goals agree with nominal hover to micrometers;
live scales verified50mmXYZ/.5radrotation, delta mode,gripper0_grip_site.

Conclusion: this supports the bounded settling handoff repair on both tested
T4seeds. It is **not grasp/task improvement**. Proceed to a new independently
identified canary with the proven settling policy; preserve paired controls,
fresh-frame retries, all guards and budgets. Do not relaunch the old immediate
opening arm or tune thresholds based on these two cases.

## Handoff probe launched; opening campaign final — 2026-09-04 00:56 UTC

**1919319** submitted00:56:11UTC; RUNNING00:56:45UTC on compute-3-12,
oneA40/eightCPUs/64GB,15minute limit. Immutable clean source
**4c3ba0b1c898cd1bf68e41256fb07ab1ce7c6930**,
label `v9d_molmo_handoff_4c3ba0b`. ExactSHA/runtime/diagnostic entrypoint
verified; model_load_count_expected0. First immediate-control case reproduced
preshape pose_drift with61total actions. No treatment result at this audit.
Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_handoff_4c3ba0b_1919319/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_handoff_4c3ba0b_1919319`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919319.out`.
BundleSHA256 `d85d19b3ac81d19d7aaf5b34df4c3fe631f3c7bc7ed3def56470129e7d1e52d6`.

**1919318 final:** opening_probe_stopped, workloadexit2, archiveVERIFIED.
Control completed **2/12**, retainedlift evidence8, successfulretries2,
3734actions,1214.304s perception; eachsuite1/6. Failuresgrasp_failed4,
no_candidates4,task_failure2. Opening40mm0/12planned with2attempted/stopped,
bothcontrollerfailures; actual118actions preserved in its helperaudits.
No improvement and no60extension. This repeats the prior full-open screen's
success/action counts; runtimes differ. Historicalmatchedbaseline remains3/12.

## Handoff diagnostic release gate — 2026-09-04

Implemented standalone `run_molmo_opening_probe.py` and isolated
`V9D_MOLMO_HANDOFF_PROBE=1` launcher mode. Normal canary/default dispatch is
unchanged. Four fresh cases record immediate versus original-target settling;
numeric controller telemetry and failure traces are retained. Exit0 means
four durable diagnostic records, not successful opening or task completion.
`treatment_passed` requires both treatment openings completed; separately
`control_reproduced` requires both expected immediate pose_drift failures.

Also fixed future early-failure reporting: valid experimental shared budget
and opening/hover audits flow into partial_audit; campaign totals prefer live
wrapper totals over stale nested retry results and count each cell once.
Missing action evidence is reported explicitly, not assumed zero. Historical
job1919318 raw artifacts are unchanged; its two opening failures used118actions.

Final combined gate **89 passed in7.89s**, independent tester **12 passed**,
focused reviewer **22 passed** and **APPROVE**, no HIGH/BLOCKER. Actual Bash
mode selection/context/dispatch/rejection tested; compilation, shell syntax,
diff check and frozen canonical-config comparison passed. One review finding
in failure-trace construction was fixed and narrowly regression-tested.
Approval covers only this bounded diagnostic. Do not deploy settling into
the production canary unless both treatment cases pass their unchanged guards.

## Opening arm stopped; bounded handoff diagnostic next — 2026-09-04

Fresh live audit **00:40:09 UTC**: job1919318 remains RUNNING; narrowed-opening
arm stopped after two distinct vanillaT4 cells (seeds1000/1001), both
`preshape_failed:pose_drift`, before perception/candidates/evaluator. Thus
**0 successes /12 planned, 2 attempted**, not an executable completed screen.
Control continues; later artifact read showed **1 success /8 terminal /12
planned**, four retention-evidence cells, one successful retry,2060actions,
692.257s perception. These control numbers are partial.

Failed helper traces show13/10shaping actions and shared totals61/57.
The current campaign incorrectly reports0actions for these early failures;
**118 total actions are verified from the two durable preshape audits**.
Preserve original artifacts; repair only future partial-audit reporting.

Pose drift reaches20.0719/20.0811mm; last available opening69.42/71.03mm.
Hover had satisfied its20mm positional tolerance but was still translating
about6mm in its last50ms policy step. Immediate shaping anchored its hold
to that transient measured pose. This suggests a handoff dynamics problem;
it does not prove a scale/frame defect. Installed OSC config is delta mode,
XYZscale50mm and rotationscale0.5rad; helper and hover use the same conversion.

Architect-approved next diagnostic: two fresh vanillaT4 seeds, immediate
preshape versus settle at the ORIGINAL nominal hover target then unchanged
preshape. No Molmo/SAM inference, candidates, grasp execution or evaluator.
Settling requires five consecutive steps with <=0.5mm displacement and
<=0.01rad orientation increment, within original20mm/.12rad hover tolerance
and >=80mm above observed regionq90; max160settlingactions/shared1200.
Keep all preshape guards unchanged. Require completed35-45mm measured opening
on BOTH settled cases before considering a production repair. If not, pivot;
no iterative threshold/gain tuning. Read robot-only controller telemetry.

## Fixed-opening paired screen launched — 2026-09-04 00:27 UTC

Job **1919318**, submitted **00:27:14 UTC**, verified **RUNNING** at
**00:27:41 UTC** on compute-3-12 (one NVIDIA A40, eight CPUs, 64GB, two-hour
limit). Exact clean execution source **97e428fcaec7030b17b8186afdb67356c81b9989**;
label `v9d_molmo_opening_97e428f`. Final combined gate **101 passed**;
independent focused review approved, no remaining HIGH/BLOCKER findings.

Ordered screen: **opening40mm first**, then **opening_control**, 12 planned
cells each; same rim_clearance prompt, hover20mm, release_plus20mm, pinned
MolmoPoint/BF16. Runtime verified torch2.10.0, Transformers4.57.1, actual A40,
matching expected/actual SHA. First arm started; no completed results yet.
Opening-probe mode is inherently screen-only; raw `screen_only=0` in job
context is an explicit mode-validation requirement, not extension permission.
No SAM, default promotion, threshold relaxation, or evaluator change.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_opening_97e428f_1919318/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_opening_97e428f_1919318`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919318.out`.
Bundle SHA256: `a08d91b286551fcd2eacb7a4de6e08aca22cb596c24f5257a4f470f79bd2268a`.

## Fixed-opening paired screen — launch-ready 2026-09-04

Validation: final combined gate **101 passed in 9.41s**, including the corrected
shell regression; independent test-only cleanup gate **43 passed in 1.70s**.
Previously 97 focused tests passed before the final launcher correction;
independent reviewer ran 47 targeted tests and approved the bounded paired
canary with no remaining HIGH/BLOCKER findings. Actual job-context rendering
under strict shell settings passed with opening mode unset, 0 and 1.
The shell heredoc correction was rechecked independently. Frozen canonical
config diff remains empty. Approval is for execution, not grasp improvement.

Next hypothesis: full-open fingers block candidate approaches that may fit with
a narrower opening. Exact local replay of the archived T4 capture reproduces
baseline0/144; translating only the two finger/pad groups to fixed40mm admits
1/144, sampled clearance6.063mm. Separate required-opening+2mm sensitivity
admits2/144, minimum6.843mm. These hypothetical poses preserve palm, box
shapes, seed/grid and original full-open robot exclusion; neither establishes
a successful grasp or actuator accuracy. All original candidates required
about15–22mm, versus observed full usable opening78.56mm.

Architect-approved production design uses NO synthetic box translation:
`opening_profile=full_open|preshape40mm`, defaultfull_open. Treatment shapes
at the converged observation hover, before fresh RGB-D, then uses actual
measured contact-pad geometry and the unchanged6mm candidate filter.
Retry shaping is after completed open/retreat and before fresh capture.
Fixed nominal40mm, accepted35–45mm; one +1 close pulse, at least5 zero-hold
actions, last3 measurements stable within0.25mm. Below35mm/missing data/pose
drift/budget exhaustion fail closed. Limit160 total shaping actions, all
charged to the shared1200. No absolute gripper command or magnitude scaling.

Runtime evidence: installed robosuite PandaGripper.format_action increments
its normalized target by sign(action)*.01 on EVERY physics substep;
SingleArm.control calls grip_action regardless policy_step. Zero holds the
target; fractional nonzero command magnitudes do not meter a smaller change.
Measured stopping/settling is therefore required.

Planned one-job paired12+12 screen: opening40mm/preshape40mm first, then
opening_control/full_open; both dense_agentview_clearance policy, rim_clearance
prompt, hover20mm, release_plus20mm, same pinned MolmoPoint and canonical
tasks/seeds/suites. No automatic60 extension. Opening change affects visible
robot configuration as well as attainable candidate geometry; this is a
whole opening-policy comparison, not an isolated static-box experiment.

Known remaining limitation: raw RGB identifies some frequent blocker pixels
as robot wrist/forearm. Current exclusion only covers the hand; full-robot
exclusion requires measured robot-owned geometry and is deferred, not mixed
into this ablation. Evaluator/object ground truth stays excluded.

## Final placement and clearance-probe results — 2026-09-03 23:53 UTC

Both jobs are terminal with workload exit0 and archive VERIFIED.
**1919241: 2/12 successes**, eight retention-evidence cells, two successful
retries, 3,593 actions, 2,058.286s perception; each suite1/6. Failures: four
no_candidates, three grasp_failed, three evaluator-rejected placements.
Visual-XY placement does not improve the observed screen over prior2/12 or
historical matching3/12. No60-cell extension or default promotion.

**1919242: diagnostic completed, not a task-success score.** Same captured
geometry yields **0 admitted /144 rejected** at BOTH6mm and5mm admission;
all are approach_obstruction. Robot exclusion is fixed6mm, seed/upper-rim
audits identical. Final controlled_stop true; first cell completed vanillaT4
seed1000 with null evaluator; motion/evaluator guards each0. Input hash
`c60a3ea83cb4bcd717587125b2f4ec9b4342c9d6bca3476bb847450579c08231`;
NPZ SHA256 `ef5651d9fdbf045b78c849ad8acaa6b995348029925053fa7c099eea39cdbd87`.
The canary's deliberate callback creates a failed-status marker internally;
the probe's final controlled-stop proof and exit0 establish diagnostic success.
This is not a failed task evaluation or candidate-motion run.

Decision: DROP the5mm-margin idea; do not launch its motion canary. Inspect
observed obstruction/contact/opening geometry for a different targeted repair.
At6mm,84first blockers are already at the pregrasp sample; at5mm,69. The
finger/hand boxes are the blocking primitives. These locate an issue but do
not alone prove an erroneous calibration or identify obstacles semantically.

## T4 perception-only diagnostic submitted — 2026-09-03

Fresh audit **23:41:06 UTC**: **1919242 RUNNING on compute-4-11**, elapsed45s.
Exact source SHA, one NVIDIA A40, BF16, torch2.10.0 and Transformers4.57.1
verified from emitted runtime record. Job1919241 also remains RUNNING.

Job **1919242** submitted at **23:39:48 UTC**, exact source
`c818ed9a59a12af0c86107f1081d1f2d5e0a7d3e`, label
`v9d_molmo_geometry_probe_c818ed9`. This is a diagnostic, not a grasp-success
canary. One A40, eight CPUs, 64GB, bounded 30-minute allocation. Bundle SHA-256
`dac9640bb09512bb0f185bccc42c105649735eb5bf70c5b74767091c5d4cc77a`
and clean immutable release were verified before submission; frozen config is
unchanged. Same-frame geometry compares 6mm versus 5mm admission, fixing robot
exclusion at 6mm. Normal observation is allowed; grasp execution and evaluator
are guarded. No claim that a smaller margin is safe or improves task success.

Launch: `V9D_MOLMO_GEOMETRY_PROBE=1`, `V9D_MOLMO_ARMS=dense_agentview_clearance`,
`V9D_MOLMO_OBSERVATION_PROFILE=hover20mm`, `V9D_MOLMO_MOTION_PROFILE=baseline`,
`V9D_MOLMO_SCREEN_ONLY=1`, repair gate and motion probe both 0;
`sbatch --parsable --time=00:30:00 vla_benchmarking/legion/v9d_molmo_campaign.sbatch`.
The probe fixes the actual cell to vanilla T4 seed1000 and rim_clearance.
Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_geometry_probe_c818ed9_1919242/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_geometry_probe_c818ed9_1919242`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919242.out`.
No automatic resume or duplicate submission. Use actual admitted counts and
final zero-guard-invocation proof to decide the next experiment.

At **23:39:53 UTC**, placement job **1919241** has **2 successes /11 terminal
/12 planned**, seven retention-evidence cells, two successful retries,
3,304 actions and 1,887.026s perception. Failures: four no_candidates, two
grasp_failed, three evaluator-rejected placements. Each suite currently 1/6.
Not yet terminal; no improvement established against historical matching 3/12.

## Bounded T4 perception replay and resume repair — preparation 2026-09-03

Final local gate:80 focused tests passed; independent tester63checks including
probe CLI/guards and actual matrix extension passed. Identity extension runs
24new cells/suite(seeds1002–1009) while preserving terminal success, evaluator
false, failure and interrupted prefix cells. Changed/legacy identity rejects
without mutating original matrix records. Shell syntax, Python compilation,
diff and frozen-config checks passed. Independent review APPROVE,36focused
checks passed independently, no HIGH/BLOCKER. Approval covers only the
perception-only probe, not a5mm grasp-motion canary or improvement claim.

Implement a separate perception-only diagnostic, not a task-success canary.
One fresh vanillaT4seed1000 cell uses normal hover20mm and rim_clearance Molmo
perception, captures complete geometry inputs, then runs pure geometry at6mm
and5mm admission margins with current-robot point exclusion fixed6mm. Return
no candidates to the motion runner and guard candidate execution/evaluator.
Persist RGB, metric depth, mask, camera K/T, robot OBB calibration, points,
policies and hashes. Admission counts—not stored first-hit clearances—decide
whether a subsequent margin-only canary is worth running. No hardware safety
claim, default change or reduced action/retention requirements.

Verified prior first-hit counts >=5/4/3mm across current T4 cells:
vanilla1000:27/39/50; vanilla1001:9/24/41;
sealed1000:34/63/81; sealed1001:33/58/75. These are NOT admission counts.
The helper exits at its first blocker, and the old margin also changes robot
exclusion/upper-rim support. Decouple these with optional robot exclusion
margin(defaultNone retains old behavior). Reject explicit collision status
at equality too: helper uses<= while caller previously used<. Both new-code
diagnostic arms share that fail-closed boundary correction. Old1919241 stays
immutable. No prior exact replay is possible: T4 artifacts lack full depth/mask
and robot-probe arrays. Do not infer object identity for offending pixels.

Parallel identity-only repair follows the architect's previous package:
scientific identity excludes operational label/phase/count/output and includes
clean execution SHA, full immutable inference/config identity, controller hash,
camera/geometry/motion/hover/task-seed-suite settings. Keep matrix resume guard
and old-prefix identities untouched. Test actual matrix extension with retained
terminal successes/failures and no replay before any future60-cell cohort.

## Visual-endpoint XY ablation running — 2026-09-03

Latest metrics **23:32:58 UTC**:10terminal/12planned,2successes,6retention-evidence,
2successfulretries,2,898actions,1,629.958sperception. Four no_candidates,
two grasp_failed,two evaluator-rejected completed placements; bothsuites1/6.
Stillrunning, below historical3/12 so no improvement claim.

Audit **23:15:17 UTC**:1919241 stillRUNNING, four terminal of12 planned,
one success(vanillaT6seed1001), two retention-evidence cells, one successful
retry,1,270actions,667.592sperception. T4bothseeds no_candidates, T6seed1000
grasp_failed; vanillaT9seed1000running. No improvement claim from this partial.

Fresh audit **23:03:40 UTC**: **1919241 RUNNING on compute-3-14**, elapsed2:19.
Exact source, one NVIDIA A40, BF16, torch2.10.0/Transformers4.57.1 and effective
clearance/hover20mm/release20_visual_xy/screen_only1 verified. Campaign started;
no result claim yet. Initial queued snapshot follows.

Job **1919241** submitted **23:00:30 UTC**, verified **23:01:03 UTC PENDING /
Priority**, no dependency. Exact source
`1973c0961e7857e7fc9f0e26831ad9b91672ac41`, label
`v9d_molmo_visualxy_1973c09`; one A40/eight CPUs/64GB. Clean immutable release,
frozen-config diff and bundle SHA256
`2800d8d8cbf9593236112707dafa6509745ea940e3f8c58cc1bef7059a40b7f6`
verified on Legion. Effective submission: dense_agentview_clearance,
hover20mm, release20_visual_xy, screen_only1, repair_gate0, motion_probe0.
No trial results or runtime verification yet. No active/pending duplicate.

Source: `/home/hjaber/EmbodimentSemantic_runtime/checkouts/v9d_molmo_1973c0961e7857e7fc9f0e26831ad9b91672ac41`.
Expected output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_visualxy_1973c09_1919241/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_visualxy_1973c09_1919241`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919241.out`.

At 22:52:46 UTC, higher-release job1919238 has 2 successes /11 terminal /12
planned cells, seven retention-evidence cells, two successful retries, 3,444
reported actions and 1,099.421 seconds perception. Both suites currently1/6;
four no-candidate, three grasp-failed and two completed-placement/task-failure
outcomes. No improvement over historical3/12 is established.

Next question: does removing the legacy lateral transfer bias improve landing?
New opt-in `release20_visual_xy` changes only transfer XY from the preserved
classical delta to destination_visual minus source_visual. Keep its legacy Z
delta, release+20mm, hover20mm, clearance prompt, model, source grasp geometry,
rotations, budgets and post-retreat evaluator fixed. Fresh canonical12-cell
screen against job1919238; no combining results across revisions or default
promotion. The observed anchors are not proven object centers; no-slip carrying
and unchanged orientation are assumptions. This is an exploratory ablation,
not a proven double-offset bug or a claim of better placement.

Verified T9 seed1000 attempt02: legacy XY contribution is(-20.3,+5.2)mm.
Pure visual XY shifts transfer targets(+20.3,-5.2)mm, preserving candidate
contact offset and all Z values. Release changes from
[0.008362646,0.229016602,0.927280029] to
[0.028662646,0.223816602,0.927280029]m. Exact camera calibration projects
these to[203.687,87.384] and[203.239,82.399]pixels respectively. These are
grip-site pixels, not bowl centers. Actual close was2.97mm from its candidate.
Do not remove only the source offset: that leaves the large common Y bias.

Launch gate passed:62 focused tests on final code, including an independent
real run_episode to _run_motion smoke completing all11 phases in49/1200
actions. Actual stage targets, omitted-default equivalence, separate candidate
release targets, release20 and rotations checked. Independent review APPROVE,
no HIGH/BLOCKER;31 focused plus6 final transfer tests passed under review.
Python compilation, shell syntax, diff and frozen-config checks passed.
This approves a fresh12-cell experiment only, not an improvement claim.

Independent operational inspection identified a prefix-to-full60 resume hash
mismatch (phase/output identity changes). No extension or provenance bypass
has been attempted. Repair that narrow resume contract before any finalist
extension; this does not block fresh12-cell screens.
Architect direction: retain matrix strict resume guard; new scientific hash
must exclude operational phase/count/output/label but include actual clean
execution SHA and controller/model/prompt/camera/motion/hover identity. Keep
operational manifest separately. Existing old-schema prefixes cannot be silently
restamped as a new-SHA cohort; preserve them and fail incompatible resumes.
Implement before the next cohort intended for60 cells. No such code is in1919241.

Next-failure triage (read-only): all eight T4 cells across1919230/1919238 rejected
144/144 proposals for `approach_obstruction` on `pregrasp_to_contact`, threshold
6mm. Current vanilla seed1000 clearance spans-6.190 to5.949mm:53negative,
94below3mm,117below5mm, three at least5.9mm. Region/depth exists, all vanilla
Molmo points passed, no aperture-exceeded outcomes; ranking never ran. Source
gate is `_hand_volume_obstruction` in grasp_candidates.py and policy threshold
in run_molmo_sam3_canary.py. This identifies the rejection gate, not a proven
geometry bug. Next bounded diagnostic: offline clearance sensitivity from saved
proposals and geometric consistency of the nearest obstacles. Do not silently
weaken collision criteria or treat lower-margin admission as safety evidence.

## Release-height ablation completed — 2026-09-03

Final audit **22:55:29 UTC**: **1919238 completed, exit0, archive VERIFIED**.
Final **2/12**, eight retention-evidence cells, two successful retries,
3,734 reported actions, perception1,196.281s; vanilla1/6, sealed_randomized1/6.
Failures: no_candidates4, grasp_failed4, task_failure2. Two formerly stalled
placements completed but failed the evaluator. This ties the previous2/12
control and remains below historical3/12. Advance the new XY screen, not a
full60 extension of this unchanged treatment. Earlier snapshots follow.

Fresh audit **22:30:23 UTC**: **1919238 RUNNING on compute-4-11**, elapsed1:02.
Exact SHA, NVIDIA A40, BF16, pinned torch/Transformers and effective
`dense_agentview_clearance / hover20mm / release_plus20mm / screen_only=1`
verified from runtime and job context. Campaign is running; no final result yet.

Job **1919238** submitted at **22:29:00 UTC**, scheduler **PENDING / Priority**
at submission. Source `0f1f90b27a447c1e916e2605ebb0bfe4f6c9fc0c`, label
`v9d_molmo_release20mm_0f1f90b`, one A40/eight CPUs/64 GB. Clean SHA-named
release and bundle SHA-256
`a43c9e23b8e635bff30012c6c791594669bd4a8da2833cf883969a412b6b2db0`
verified on Legion. Only one older experiment remained allocated at submission;
no duplicate or running source was changed. Runtime is now verified above.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_release20mm_0f1f90b_1919238/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_release20mm_0f1f90b_1919238`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919238.out`.

Implement `release_plus20mm` against parent source
`15789cd6d0c14e2c99dee6542e33dc119857ba47`. Fixed world-Z +20 mm applies only
to release/open and retreat; keep preplace, lift, candidate orientation, XY
landing target, all grasp criteria and post-retreat evaluator unchanged.
Compare a fresh complete 12-cell clearance-prompt/agentview/hover20mm screen
with matching job1919230 cells. No SAM, no new prompt/model, no per-cell tuning.
This tests release height, not a proven contact diagnosis. It may bounce/miss
and cannot repair preplace stalls; evaluator success is the decision metric.

Matched placement-pulse evidence (job1919194, vanilla T6 seed1000): identical
candidate `seed03_yaw+15.0_ins00.0mm`, same candidate-array hash and identical
trajectory through preplace step37. One outward 5 mm target bias on steps38–45
reduced nominal residual from 26.162 mm to 23.745 mm, then returned to 26.225 mm
versus control26.229 mm. Both failed preplace after160 steps and301 total actions.
EEF force norm changed only5.1560→5.1627 N; no meaningful load spike. This was
a target-bias maneuver, not a force pulse. Sustaining the same tiny bias has
weak prospects of reaching the unchanged15 mm tolerance. The observed
~18 mm upward placement-descent residual instead motivates the height test.

Additional matched T9 seed1000 evidence: identical `seed00_yaw+00.0_ins00.0mm`
candidate and pre-pulse trace; both preplace phases reached in36 steps. A single
eight-action,5 mm downward/outward bias during descend_place briefly changed
nominal residual18.589→17.841 mm, then final residual21.578 mm versus
control21.555 mm. During the pulse, EEF force norm64.57 N versus matched
control53.47 N (+20.8%), torque3.423 versus2.939 (+16.5%). After restoration,
load and residual returned near control. This supports a release/support-contact
hypothesis for that descent, not a general explanation for preplace stalls.
Only EEF force/torque telemetry was available; no hidden object state accessed.

Saved clean/lift/preplace-timeout frames for new-hover vanilla T9 seed1001
visibly show the bowl moving from stove source to the preplace region. This
supports carrying in that individual trial beyond the proprioceptive heuristic,
without identifying the cause of the stall.

Backend owns the candidate-only release offset and workspace/audit checks;
ML worker owns profile resolution and provenance; root owns campaign/launcher.
Validation:136 tests passed, one skipped; independent real run_episode→motion
helper smoke completed all11 phases in49/1200 actions with only release/retreat
Z shifted. Independent review found no HIGH/BLOCKER and ran37 targeted tests.
Python compilation, launcher syntax, diff checks and frozen-config regression
passed. Use a new immutable release and the now-free single-A40 slot. Keep
prior releases unchanged. This approves the experiment, not performance.

## Observation-only hover repair completed — 2026-09-03

Fresh audit **22:25:14 UTC**: job **1919230 completed, exit0, archive VERIFIED**.
Exact source, one NVIDIA A40, BF16 and pinned runtime versions verified.
Final screen: **2/12 successes**, eight retention-evidence cells, two successful
retries, four no-candidate outcomes and six grasp/placement failures. All
12 observation hovers passed. Both suites1/6. Reported actions4,011;
perception1,303.645 s. This is below historical3/12 despite eliminating the
two old observation failures and doubling retention-evidence cells from4 to8.
Prioritize the new release-height screen before spending48 more cells on this
still-below-baseline profile; no full60 extension has been launched.
The dependency was fulfilled after repair4 ended and its archive was verified.

Verified observation repair evidence: vanilla T9 seed1001, formerly a hover
timeout, completed all three hover phases in 37 actions. Final measured
clearance 89.566 mm above observed q90, position error 17.628 mm, orientation
error 0.04697 rad. It then closed and passed the retained-lift heuristic,
but timed out at preplace (26.856 mm / 0.12802 rad). T9 seed1000 also retained
and reached preplace but timed out at placement descent (18.402 mm / 0.00381
rad). Thus observation execution improved; task performance is not yet improved.

Job **1919230** submitted **21:56:51 UTC**, source
`15789cd6d0c14e2c99dee6542e33dc119857ba47`, label
`v9d_molmo_hover20mm_15789cd`. Scheduler confirmed **PENDING / Dependency**
on `afterany:1919168`: it will wait for the six-arm allocation to finish,
keeping at most two experiment allocations concurrent. No cancellation or
mutation of either running release. New source is a clean SHA-named checkout;
transfer bundle SHA-256
`322d14f9c4f5b6b12b0b2d580cc9f8e2aa0c7e03602399d58347e8ac311bf4ba`
verified on Legion. Initially dependency-queued, now completed as confirmed above.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_hover20mm_15789cd_1919230/results`.
Archive target: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_hover20mm_15789cd_1919230`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919230.out`.
The existing ten-minute continuation now tracks this job too; never duplicate it.

New `hover20mm` observation profile is implemented and tested in the
experimental worktree, parent `00a3b983f02984f2f92dc6cb3bdb30b11414755c`.
Hypothesis: a 20 mm observation-only positioning tolerance can avoid the
observed 15.699 mm hover stall while using the actual synchronized capture
transform. This is not a relaxation of grasp, placement, or evaluator success.
Require measured EEF height at least 80 mm above observed upper-support q90,
unchanged 0.12 rad orientation tolerance, 160 steps/phase and 1,200 actions/cell.
All three hover phases and all tasks/seeds/suites use the same policy.

Run only `dense_agentview_clearance` with unchanged `rim_clearance` prompt,
baseline placement motion, fixed model and RGB-D geometry. Use a new immutable
release and fresh complete 12-cell prefix; never combine old-policy successes
with new-policy cells. Initially screen-only, with no automatic extension.
Both current allocations remain untouched; the dependency waits for a slot under
the two-job cap. Validation: 125 tests passed, one skipped; independent actual hover/motion
helper smoke reached all phases using 3/1200 actions, 100 mm measured clearance
and zero orientation residual. Independent focused review APPROVE with no
blocking defects; 39 tests also passed under review. Python compilation,
launcher syntax and diff checks passed. Frozen v9d config diff is empty.
This establishes launch readiness, not physical improvement.

## Placement motion probe stopped and archived — 2026-09-03

Final verified at22:52:46 UTC: both arms0/12, four retention-evidence cells and
3,387 reported actions each. Control perception1,996.497s; burst2,027.902s.
Each had four no-candidate, six grasp-failed and two controller-failure cells.
Campaign `motion_probe_stopped`, exit2, archive VERIFIED. The repeated
observation-hover failure triggered its operational stop; no pulse benefit.
The following submission/partial notes are retained as historical snapshots.

Job **1919194** was submitted at **21:10:39 UTC**. Fresh scheduler audit at
**22:12:34 UTC**: **RUNNING on compute-3-14**, started 21:11:21 UTC.
New immutable source is
`00a3b983f02984f2f92dc6cb3bdb30b11414755c`, label
`v9d_molmo_placement_probe_00a3b98`. Bundle SHA-256
`3a45f86c2fb3ad14cf961ab7fa99918f80d3d7b803c9f2773a3d3b04c97ac117`
and clean exact release were verified on Legion. One A40 / eight CPUs / 64 GB;
runtime GPU/package/exact-SHA checks passed. The control arm finished **0/12**,
four passed retention heuristics. Treatment `placement_burst5mm` is running:
**0/4 terminal cells of 12 planned**, one retained-lift heuristic; vanilla
T9 seed1000 running. No treatment
benefit is established yet. Parent source is repair4
`d9c919c8e0c027554c381c0c93990465e339f251`. The current six-arm campaign remains
unchanged. No new controller/frame defect is verified. The observed placement
stalls motivate a paired response test using the existing micro-correction law.

Two dense-agentview arms, `placement_control` and `placement_burst5mm`, use
the same normal 12-cell prefix (tasks 4/6/9, seeds 1000/1001, both suites),
same prompt/model/perception/geometry, and identical opt-in motion diagnostics.
Only the latter enables correction in `preplace` and `descend_place`:
10-step plateau window, 0.1 mm movement threshold, at most 5 mm added target
bias, gain 1, eight-action bursts, one round per phase, at most 16 correction
actions per cell across retries. Corrections remain inside the original
160-step phase / 1,200-action cell budgets and workspace bounds. Success still
requires the original waypoint tolerance and unchanged post-retreat evaluator.

Question: does a small extra command cause progress toward the nominal target?
No response plus increased measured load supports a physical constraint;
without reliable load telemetry, nonresponse is inconclusive. This is a
diagnostic ablation, not a validated fix. It is screen-only, never extends to
60 automatically, and will use a new immutable SHA release after focused tests
and one independent review. Root owns campaign/SBATCH wiring, ML worker owns
profile/provenance, backend worker owns opt-in diagnostics and safety gates.
Validation: 114 targeted tests passed, one skipped; independent real-helper
smokes for open/hover/recovery with diagnostics enabled passed. One HIGH
diagnostics keyword mismatch found in the focused review was fixed and covered
by a regression test. Final independent review PASS, no remaining HIGH/BLOCKER.
This approves the bounded diagnostic launch, not a claim that placement is fixed.

Expected results: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_placement_probe_00a3b98_1919194/results`.
Expected archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_placement_probe_00a3b98_1919194`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919194.out`.
The ten-minute continuation now tracks both jobs and requires examining the
probe's actual response/load evidence before the next repair. Do not duplicate
either queued or running job. No default change or automatic probe extension.

First diagnostic evidence: control vanilla T6 seed1000 preplace timed out at
160 actions with 26.229 mm residual. The recorded controller goal agrees with
the nominal target ([0.044548, 0.252117, 1.034205] m) while nonzero position
commands continue. Controller EEF/goal and EEF force/torque readings are
available; requested/applied joint torque is not exposed. Terminal force was
[-0.2122, -0.5419, -5.1231] in the exposed sensor convention. This is not yet
evidence selecting a controller-vs-contact cause; await the matched pulse arm.

Observation-only failure evidence: control vanilla T9 seed1001 hover-translate
timed out at 160 steps with 15.699 mm residual against a 15 mm tolerance.
Target [0.087130, -0.236302, 1.276899] m, actual EEF
[0.077091, -0.233433, 1.265175] m. The controller goal agrees with the nominal
target. Assess a separately versioned observation acceptance/servo repair;
grasp/placement convergence must not be loosened and active releases stay fixed.

## Experimental repair4 completed and archived — 2026-09-03

Fresh audit: **2026-09-03 21:58:55 UTC**. Job **1919168** has exited code 2;
archive status **VERIFIED** with matching source. All six 12-cell screens are
terminal. Campaign status `no_successful_arm` means no eligible finalist, not
zero task successes: clearance scored 2/12 but failed the operational gate.
It ran on compute-4-11, one NVIDIA A40, submitted 20:09:00 and started 20:09:20 UTC.
Runtime GPU, package and exact-source checks passed. Source is immutable
`d9c919c8e0c027554c381c0c93990465e339f251`, label
`v9d_molmo_repair4_d9c919c`. Dense-agentview screen is terminal: **0/12
successes**, with **4 explicitly passed retention heuristics**, four
no-candidate outcomes, six grasp failures and two controller failures. Both
suites scored 0/6. It ended with return code 2 after the second hover-translate
160-step timeout, with all 12 planned cells recorded; this is not a fatal
campaign-contract error. Reported actions: 3,387 (incomplete for pre-audit
hover failures); perception time: 1,158.16 s; arm elapsed: 1,357.93 s.
It cannot qualify for extension and is below both matching historical baseline
prefixes (3/12 each, different historical source commits).

Geometry-only, local and contact agentview also finished **0/12 each**, with
four, three and four retention-evidence cells respectively. Each stopped after
the second hover-translate timeout on its final planned cell. Contact arm had
one completed placement/retreat sequence with an unsuccessful evaluator result.
The clearance-prompt arm has the first **two evaluator-confirmed successes**:
**2/12 terminal trials, 12 planned**. Both
successes completed placement and retreat before evaluation:

- Vanilla task6 seed1001: 606 total actions, third candidate
  `seed02_yaw-15.0_ins00.0mm`; yaw+15 and yaw+00 had timed out during translation.
- Sealed task6 seed1000: 208 actions, first candidate
  `seed01_yaw+15.0_ins00.0mm`.

Both are cells where both historical baselines failed. This is meaningful
candidate/retry execution evidence, **not aggregate improvement over v9d**:
the matching historical full prefix is 3/12 and this arm finished 2/12.
Do not bypass its repeated-hover operational stop rule for extension.
Wrist finished **0/12**, no retention evidence, with four no-candidate outcomes,
three grasp failures, three controller failures and two recovery failures.
Across six arms: **72 terminal trials, two successes, 19 retention-evidence
cells**. No finalist was extended. All raw results remain in the verified archive.
The separate placement probe and new hover-policy screen continue.
The frozen baseline and original checkout are untouched.

Bounded read-only placement audit: vanilla T6 seed1000 passed a retention
heuristic but timed out at preplace with 26.23 mm position residual. Vanilla
T9 seed1000 reached preplace in 36 steps, then timed out during placement
descent: release target Z 0.901941 m, last observed EEF Z 0.920164 m,
21.56 mm total position residual. Shared transfer height is
`max(source.z,target.z)+0.08 m`; the evidence does not establish insufficient
preplace height or a frame bug. Contact-limited descent/controller convergence
is a future hypothesis, not a verified cause. Keep this run unchanged and
collect the remaining screens before choosing the next repair.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_repair4_d9c919c_1919168/results`.
Archive target: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_repair4_d9c919c_1919168`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919168.out`.

Repair4 tests **observed upper-rim contact selection**: use actual RGB-D support
within 8 mm below its world-Z 90th percentile, with 15 mm metric neighborhoods
for tangent and per-yaw opening. Do not lift low image-boundary points to an
invented height. Preserve the repair3 hover, collision boxes, motion limits,
model/prompt input, seeds, suites and evaluator. A separate reporting correction
preserves an explicitly passed lift-retention heuristic when placement later
fails; it does not change motion or task-success accounting. The temporary
global four-cell gate is disabled: shared motion executed successfully in
repair3, and a physical no-candidate outcome must not block other camera/prompt
arms. All six 12-cell screens run sequentially, retaining per-arm operational
stop rules and immediate fatal-contract stopping. Only complete arms with at
least one successful placement may advance to 60 cells (at most two).
All-zero screens are archived for a new idea instead of extended. Targeted
validation passes 44 tests plus independent focused testing and review.
Independent read-only audit of installed Robosuite 1.4.0 confirmed that
`robot0_eef_pos` and OSC `ee_pos` use the same `grip_site`; the body quaternion
is explicitly converted using the measured body-to-site rotation. No verified
position-frame mismatch explains the residual motion errors. Do not treat that
discarded hypothesis as a reason to change frames. Placement/transit dynamics
and contact feasibility remain runtime questions.

## Experimental repair3 finished — 2026-09-03

Fresh scheduler audit: **2026-09-03 19:57:05 UTC**. Job **1919163** is
**FAILED, stopped_repair_gate, exit 2**, ran 19:45:20–19:53:15 UTC on one A40.
All four hovers completed. T4 seeds 1000/1001 had no candidates (108/36 and
36/108 obstruction/aperture rejections). T6 executed six grasp candidates;
seed 1000 reached close/lift on attempt 2 then timed out during preplace;
seed 1001 timed out during three translations and one pregrasp. Final result:
**0/4 task successes**, 1,509 total reported actions across four cells, no
other arms or finalists run. The archived metric says zero retained lifts,
but direct phase evidence shows **one passed retention heuristic** before the
placement failure (qpos 0.001786/-0.002016, threshold 0.0015). This is not
verified object retention. The reporting bug will be corrected in repair4.
Archival is VERIFIED; no Molmo job is active. Unrelated job 1918812 is untouched.
Executed source is immutable commit
`4ae6be0b4da7fb12731fbc6d7796433794a13566`, label
`v9d_molmo_repair3_4ae6be0`. The source bundle hash was verified on Legion,
and the deployed SHA-named checkout is clean. Independent final review
approved the gated launch with no HIGH/BLOCKER findings.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_repair3_4ae6be0_1919163/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_repair3_4ae6be0_1919163`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919163.out`.

The user authorized continued evidence-driven fixes and Legion relaunches
without further confirmation until improvement. A thread follow-up checks
every ten minutes and continues work; it must not duplicate active jobs.

Repair3 changes only the shared observation route/height and robot hand-volume
representation: rise, translate, then lower above upper observed RGB-D support;
replace coarse collision spheres with oriented bounds enclosing the actual
robot collision geometry. Clearance, aperture, action limits, candidate policy,
model, seeds, evaluator and simulator remain unchanged. This is an exploratory
combined integration repair, not a prompt-effect estimate. Use the four-cell
repair gate before allowing the existing six-arm screen and finalist extension
in that historical run. Repair4 supersedes only this temporary global gate.
Pre-launch targeted validation: **39 passed** across geometry, episode adapter,
campaign and frozen-controller isolation tests. Independent tester confirmed
rotated-box rejection, malformed geometry rejection, staged hover execution and
phase bounds. An existing timeout-message discrepancy is diagnostic only; the
configured action bound is enforced and both limits are 160 in this campaign.

Historical matching-cell comparison was verified from archived matrix files:
both jobs **1914768 and 1917528 scored 3/12** on tasks 4/6/9, seeds 1000/1001,
both suites (vanilla 1/6, sealed_randomized 2/6). Both had zero successes on
the four vanilla T4/T6 repair-gate cells. Historical source identities are
`06aa8fe7e520549b550ee5f7507237968001776c` and
`d32e72ffbe0b905af1e1dc14b6afe53aaa487472` respectively, with the same canonical
v9d configuration hash `60f4f5f9ecfde7b4830f376ab06cfc706e2ef175d86817c42a0adb7cddd46c0c`.
The complete historical scores remain 15/60 and 14/60. None is a same-commit
comparison with the experimental treatments. Every failed revision remains
in the records below and in its immutable archive.

## Experimental v9d + Molmo repair campaign — 2026-09-03

Scoped status audit: **2026-09-03 16:51:05 UTC**. Legion job **1919010** is
**finished, `stopped_repair_gate`, exit 2**; it ended at 16:02:43 UTC on
compute-3-12 (one NVIDIA A40). The first dense-agentview arm recorded **zero
successes and retained lifts in four terminal trial cells**. No grasp candidate
was executed. T4 seeds 1000/1001 completed hover, but all 144 candidates per cell
were rejected (90/36 obstruction rejections and 54/108 aperture rejections).
Molmo returned 4/3 points respectively. Both T6 hovers exceeded 160 steps,
ending 34.9/58.7 mm from their targets. The remaining five arms and finalist
extensions did not run. Archival is VERIFIED. Runtime CUDA/package/exact-commit
checks passed, but usable physical motion remains unproven.
Executed source is immutable commit
`709a29bdf473e2a7edb255da5cd33430a119cfe0`, label
`v9d_molmo_repair2_709a29b`. It uses the v9d RGB-D region backend and pinned
`allenai/MolmoPoint-8B` revision `188130f961c8e0888a34e11121a1423c461a01ba`,
without SAM. The frozen baseline remains
`fd24a4c5cf8da4991013ab18b15704523ad0836b`.

Repairs preserve explicit candidate pregrasp, forward v9d motion settings,
stage rotation at clearance, restore immediate empty-close detection, refresh
robot calibration before proposals, rank using live pose with stable jaw-symmetry
identities, and check observed hand volumes. **34 targeted tests passed** and
the focused independent review approved the gated launch.

The single A40 job first runs the existing four vanilla T4/T6 cells (seeds
1000/1001). It continues only when all observation hovers complete and each task
has at least one completed close/lift. It then completes the same six-arm
12-cell screens and extends the best two eligible arms to 60 cells each without
repeating matching cells. This is a runtime integration gate, not evidence of
task success. No default promotion is authorized.

Output: `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/v9d_molmo/runs/v9d_molmo_repair2_709a29b_1919010/results`.
Archive: `/home/hjaber/EmbodimentSemantic_archive/v9d_molmo/v9d_molmo_repair2_709a29b_1919010`.
Log: `/home/hjaber/EmbodimentSemantic_runtime/operator/logs/v9d_molmo_campaign_1919010.out`.

Previous job `1918783` at `1a42f70ac1270dcb949b161a9e7e517238699f93` ended
`no_executable_arm`: 24 terminal cells out of 72 planned screen cells, zero
successes and retained lifts. Its shared motion failures cannot rank prompts.
Its raw results remain archived. These records exist only in the experimental
worktree; the original checkout's user-edited ledgers were not changed.

This is the human-readable source of truth for the active SmolVLA LoRA
experiments. Its machine-readable companion is
`evaluation_results_tracker.json`. **Both files must always describe the same
state.**

The always-applied startup rule enforcing this requirement is
`.cursor/rules/smolvla-experiment-updates.mdc`.

**Analysis scope:** epoch 15 checkpoints only (`029190`, 29,190 steps). Do not
add any other checkpoints unless the user explicitly changes the scope.

**Last active-run audit represented in both files:** 2026-08-31 08:37:13 UTC.
The active forward run is on PoliTO Legion. The last preserved Lambda-only audit
is 2026-08-20 10:11:17 UTC; its process IDs are historical snapshots, not live
state.

## Mandatory rule whenever the user asks for an update

Before answering any request such as "updates?", "where are we?", "how many?",
"is it done?", or "results so far?", the active thread must complete this exact
transaction:

1. Connect to each active execution host and refresh every running training and
   evaluation job. Current forward runs use Legion through the mp4 gateway.
2. Inspect process state, the latest training log/checkpoints, completed episode
   artifacts, and `eval_info.json` when present.
3. Update `evaluation_results_tracker.json`, including `audited_at_utc`, job
   status, progress, PIDs, per-task counts, and whether each score is partial or
   final.
4. Update this ledger with the identical audit time and identical facts.
5. Parse the JSON and check both files for contradictory statuses or scores.
6. Only after steps 1–5, answer the user with the new numbers.

If an active execution host cannot be reached, do not present its snapshot as
current. Record and report that the refresh failed, retain the last successful
audit timestamp, and label every shown count stale.

## Terminology that must not be mixed up

Never use the word **control** by itself. It has referred to two different
things:

- **Frozen base:** the pinned `HuggingFaceVLA/smolvla_libero` policy with no LoRA
  fine-tuning.
- **No-arrow LoRA:** a LoRA fine-tuned for 15 epochs on normal LIBERO images with
  no arrows. Its dataset directory happens to be named `control`, but this model
  is not the frozen base.

Every result must explicitly state both:

- **Trained on:** the image condition baked into the fine-tuning dataset.
- **Evaluated with:** the live overlay shown during LIBERO rollout.

Image conditions:

- **All arrows:** multiple arrows from `akita_black_bowl_1` to selected visible
  spatial-relation objects.
- **Target arrow:** exactly one arrow from `akita_black_bowl_1` to the task goal,
  `plate_1`.
- **No arrows:** no synthetic arrow overlay.

## Shared experimental contract

- Base revision: `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
- LoRA rank: 16
- Batch size: 32
- Seed: 1000
- Epoch-15 checkpoint: `029190`
- Save interval: 1,946 steps, one checkpoint per epoch
- Evaluation cameras: `agentview,robot0_eye_in_hand`
- Observation size: 256x256
- Main checkpoint probe: tasks 0 and 7, 10 episodes per task
- Task 0/7 environment variation: canonical LIBERO initial-state variation and
  existing static prompt overrides. These tasks do not receive this repository's
  custom object swaps, removals, or camera-selection interventions. Do not call
  these probes custom randomized-scene evaluations.

## Exact training runs

### A. All-arrow LoRA — epoch 15 selected

- ID: `all_arrows_lora_epoch_15_checkpoint`
- Trained on: all arrows baked into the training images
- Status: complete
- Selected checkpoint:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/treatment_2026_08_17_19_42_03/checkpoints/029190/pretrained_model`
- Dataset:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_datasets/treatment`
- Dataset size: 500 episodes, 62,250 frames
- The source run later continued beyond epoch 15. This ledger uses only
  checkpoint `029190`.

### B. Target-arrow LoRA — 15 epochs

- ID: `target_arrow_lora_15_epochs`
- Trained on: exactly one baked bowl-to-plate target arrow
- Status: complete
- Final checkpoint:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/target_arrow_treatment_2026_08_18_19_47_55/checkpoints/029190/pretrained_model`
- Dataset:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_datasets_target_arrow/target_arrow_treatment`
- Dataset size: 500 episodes, 62,250 frames

### C. No-arrow LoRA — 15 epochs

- ID: `no_arrows_lora_15_epochs`
- Trained on: normal LIBERO images with no arrows
- Status: complete, 29,190 / 29,190 steps
- Run directory:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/no_arrow_treatment_2026_08_19_13_08_20`
- Expected final checkpoint:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_runs/no_arrow_treatment_2026_08_19_13_08_20/checkpoints/029190/pretrained_model`
- Dataset:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/lora_datasets/control`
- Dataset size: 500 episodes, 62,250 frames
- Launcher PID: not running (not a long-lived process)
- Active training PID: not running
- Active worker PIDs: none
- Latest checkpoint observed: 029190
- Log:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/no_arrow_treatment_launch_2026_08_19_13_08_20.log`

## Completed final evaluations

### 1. All-arrow LoRA evaluated with all arrows

- Model: all-arrow LoRA
- Trained on: all arrows
- Evaluated with: all live arrows
- Checkpoint: epoch 15 (`029190`)
- Task 0: 6 / 10 successes
- Task 7: 5 / 10 successes
- **Final score: 11 / 20 (55%)**
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/checkpoint_probe_029190_tasks_0_7_treatment`

### 2. Frozen base evaluated with all arrows

- Model: pinned frozen base, no LoRA
- Trained on: not fine-tuned
- Evaluated with: all live arrows
- Task 0: 0 / 10 successes
- Task 7: 0 / 10 successes
- **Final score: 0 / 20 (0%)**
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/checkpoint_probe_base_tasks_0_7`
- Critical clarification: this is not a frozen-base/no-arrow result.
  `visual_relation_audit.jsonl` records `condition=visual_arrows`.

### 3. Target-arrow LoRA evaluated with one target arrow

- Model: target-arrow LoRA
- Trained on: one target arrow
- Evaluated with: one live target arrow
- Checkpoint: epoch 15 (`029190`)
- Task 0: 4 / 10 successes
- Task 7: 1 / 10 successes
- **Final score: 5 / 20 (25%)**
- Evaluation PID: not running (complete)
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/eval_outputs/target_arrow_epoch15_tasks_0_7`
- Overlay evidence: `visual_relation_audit.jsonl` records
  `visual_goal_arrow` and one bowl-to-plate relation.

### 4. All-arrow LoRA evaluated without arrows

- Model: all-arrow LoRA
- Trained on: all arrows
- Evaluated with: no arrows
- Checkpoint: epoch 15 (`029190`)
- Task 0: 5 / 10 successes
- Task 7: 0 / 10 successes
- **Final score: 5 / 20 (25%)**
- Evaluation PID: not running (complete)
- Output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/checkpoint_probe_029190_tasks_0_7_treatment_no_arrows`
- Overlay evidence: no `visual_relation_audit.jsonl` is expected because
  `VISUAL_CONDITION=none`.

## Historical Lambda-only evaluation snapshot

### 6. No-arrow LoRA evaluated without arrows

- Trained on: no arrows
- Evaluated with: no arrows
- Planned checkpoint: epoch 15 (`029190`)
- Planned coverage: task 0 and task 7, 10 episodes each
- Status: **stale historical snapshot; not a current running-state claim**.
- Last observed PID on Lambda: 56554 at the preserved 2026-08-20 audit.
- The active forward run moved to Legion, and this Lambda process was not
  refreshed during the current Legion-only audit.
- Planned output:
  `/home/ubuntu/EmbodimentSemantic/vla_benchmarking/eval_outputs/no_arrow_treatment_epoch15_tasks_0_7`

## Verified missing baseline

The **frozen base evaluated without arrows** has not been run. No matching
`eval_info.json` exists on Lambda. The frozen-base 0/20 result above was
evaluated with all arrows.

This is different from the no-arrow LoRA evaluation: the frozen base has no
fine-tuning; the no-arrow LoRA receives 15 additional training epochs.

## Aborted attempts that are not results

These training directories produced no adapter checkpoint and must never be
reported as trained models:

- `pair_2026_08_17_18_46_29`
- `pair_2026_08_17_19_18_06`
- `pair_2026_08_17_19_21_06`
- `treatment_2026_08_17_19_40_32`

These evaluation directories produced no videos or `eval_info.json` and must
never be reported as results:

- `intermediate_eval_029190_live_arrows`

Do not delete these or any other Lambda artifacts without explicit user
authorization.

## Completed Legion no-arrow causal-control experiment

Audit time: **2026-08-30 20:19:37 UTC**.

### Training — `legion_no_arrow_lora_full_s1000_v1`

- Status: **complete** on PoliTO Legion.
- SLURM job: `1910197` on `gpu_a40`; it has left the live queue.
- Source commit: `8579b62e58aad28e131a8b8da370b4c34f2fc013`.
- Trained on: **no arrows**, using the no-arrow half of the exact sealed
  `sealed_lora_control_treatment` pair.
- Base revision: `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- Schedule: 15 epochs, 29,190 steps, checkpoint every 1,946 steps, batch 32,
  seed 1000, LoRA rank 16.
- Final checkpoint: `029190`; all 15 scheduled checkpoints exist.
- Completion evidence: step 29,190 and `End of training` at 08:27:49 UTC,
  adapter postcondition and reload smoke passed, launcher finished at 08:28:09
  UTC.
- Final adapter SHA-256:
  `80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092`.
- Training-manifest SHA-256:
  `95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391`.
- Scratch run:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197`
- Durable archive:
  `/home/hjaber/EmbodimentSemantic_archive/runs/legion_no_arrow_lora_full_s1000_v1_no_arrow_treatment_1910197`
- Score: training itself has no rollout score; final evaluation results are
  recorded below.

The refreshed stage, no-arrow preflight, and two-step A40 smoke completed before
this submission. The smoke wrote and reloaded a no-arrow adapter successfully.

### Evaluation — `legion_no_arrow_trained_live_vs_none_s1000_ep10_v1`

- Status: **complete**, SLURM job `1910198`; it has left the live queue.
- It evaluates the same final no-arrow-trained adapter in exactly this order:
  1. `no_arrow_trained_live_arrows` — trained on no arrows, evaluated with live
     all-object arrows.
  2. `no_arrow_trained_no_arrows` — trained on no arrows, evaluated without
     arrows.
- Coverage per cell: tasks 0–9, 10 episodes per task, seed 1000, batch size 1.
- Randomization and prompt configuration: unchanged sealed current config.
- No frozen-base or arrow-trained model enters this evaluation job.
- Scratch evaluation root:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/eval/legion_no_arrow_trained_live_vs_none_s1000_ep10_v1_no_arrow_treatment_1910198`
- Final scores from each cell's authoritative `eval_info.json`:
  - `no_arrow_trained_live_arrows` — trained on no arrows, evaluated with live
    all-object arrows: tasks 0–9 = `8, 4, 3, 4, 0, 0, 7, 1, 3, 0` successes
    out of 10; **30/100 (30%) final**.
  - `no_arrow_trained_no_arrows` — trained on no arrows, evaluated with no
    arrows: tasks 0–9 = `9, 9, 6, 4, 1, 1, 6, 4, 3, 0` successes out of 10;
    **43/100 (43%) final**.
- Cell completion times were 10:16:18 UTC and 11:45:18 UTC, respectively.
- Durable archive:
  `/home/hjaber/EmbodimentSemantic_archive/eval/legion_no_arrow_trained_live_vs_none_s1000_ep10_v1_no_arrow_treatment_1910198`.
- The final adapter, training manifest, both `eval_info.json` files, and the
  pair-summary CSV have identical SHA-256 hashes in scratch and durable HOME
  storage.

## Graph-text training setup attempts

- Source commit for the final launcher: `e83e3f5e4b7f97b817366acbd89af021001de829`, pinned in the clean
  Legion graph checkout.
- Repair: the graph snapshot manifest excludes runtime `.cache` metadata, and
  the run uses a new immutable `-graph96-v2` snapshot path so the invalid legacy
  `-graph96` artifact is preserved rather than overwritten.
- Setup/smoke job `1911386` **failed at `2026-08-30T20:45:23Z`** after the
  graph-pair verification gate; training and evaluation were not submitted.
  The failure was a stale historical-pair sentinel digest in the existing
  derived graph manifest, not a source-data mismatch.
- Derived-artifact repair job `1911400` completed `0:0` at `21:37:06Z` after
  rebinding and fully verifying the graph-pair sentinel. A backup of the stale
  manifest is retained beside the repaired derived artifact.
- Fresh setup/smoke job `1911425` failed at `22:01:02Z` because its historical
  verify pass deleted the graph sentinel before the stale-hash check; no
  training/evaluation was submitted. A subsequent `1911444` attempt failed
  closed on the now-missing sentinel in 3s.
- Standalone repair verification job `1911451` completed successfully by log
  evidence at `22:52:02Z`: it verified 62,250 frames and 500 episodes, rewrote
  the graph-pair sentinel, and produced no stderr.
- The newest setup/smoke job `1911474`, using source commit `e83e3f5e4b7f97b817366acbd89af021001de829`,
  passed the graph-policy audit but failed at `23:17:40Z` because `job_dir` was
  unbound at line 146 of the generated SLURM script. Its setup archive is
  verified (`78cfafad7f44beef6b75f6dae5235af8199c49e5d92db4941eae1f675fcf2e3c`).
  The state file records `setup_status=FAILED`, `input_bundle_status=FAILED`,
  and no training or evaluation job IDs.
- Repaired setup/smoke job `1912529` completed successfully at
  `2026-09-01T09:49:34Z` from source commit
  `6a717c91a80d7a201ae99bd02d46d37151e3f701`. The existing graph pair passed
  immutable preflight, and the existing `graph96-v2` base snapshot was reused;
  no dataset regeneration or overwrite occurred.
- The repaired setup passed the 2-step GPU LoRA smoke (checkpoint `000002`),
  adapter reload, live checkpoint smoke for tasks 0 and 2, and the real
  terminal-success reset smoke. The previous list/`.ndim` failure was fixed by
  coercing the dummy action to a NumPy `float32` array before `LiberoEnv.step`.
  The setup archive is verified with tree hash
  `bad36313db6f05d60c3424edf7787bb8a63afe819fe28df66bb31f79906f4380`.
- After explicit launch confirmation, training job `1912720` completed and its
  paired evaluation job `1912721` started on `compute-4-11`. The verified setup
  state and sealed templates were reused; no second setup or dataset
  regeneration was performed.
- Fresh Legion audit: `2026-09-02T02:47:32Z`. Training reached step `29,190`
  / epoch `15`, saved all 15 checkpoints through final `029190`, passed the
  adapter reload smoke, and finished at `01:50:19Z`; its archive is verified.
  Evaluation `1912721` is still **RUNNING**, currently in the
  `graph_trained_graph_context` cell. Six task video artifacts (tasks 0–5) are
  present so far; final success metrics are not available yet.
- At the fresh `2026-09-02T02:52:11Z` audit, the evaluator still had no
  complete `eval_info.json` or `randomization_audit.jsonl` rows. Therefore no
  per-task success count is reported yet; the existing task 0–5 artifacts are
  partial debug/video outputs, not evaluation results.
- Trained on: graph-text `target_natural_v1` with no visual arrows, using the
  sealed `graph_treatment`/`arrow_graph_treatment` pair.
- Planned contract: 15 epochs, 29,190 steps, checkpoint `029190`, batch 32,
  LoRA rank 16, seed 1000; paired evaluation uses graph-present and
  graph-removed text with no visual arrows, 10 episodes per task for tasks 0–9.
- The failed-attempt state file above is historical. The current verified setup
  state is `/home/hjaber/EmbodimentSemantic_runtime/graph_pilot/legion_graph_treatment_lora_full_s1000_v1_20260901T091457Z/state.env`.
- The setup smoke checkpoint `000002` is diagnostic only. The full-training
  final checkpoint is `029190`; no evaluation score exists yet. The old failed
  chain's jobs `1911247` and `1911248` remain dependency-blocked and are not
  current training progress.

The previous setup job `1911381` was cancelled before completion because it was
still targeting the stale derived snapshot. It produced no training/evaluation
submission; its partial setup evidence remains archived. The follow-up setup
`1911382` was also cancelled before completion after the launcher-state contract
fix was identified; it produced no training/evaluation submission and must not
be reused. Failed setup `1911386` is likewise retired after exposing the stale
historical-pair sentinel. Setup `1911474` is the latest attempt and is terminally
failed; setup `1912529` is the later verified setup. Training `1912720` is
complete and evaluation `1912721` is the active dependent evaluation.

## Completed action-visual training — `legion_action_visual_lora_no_arrow_s1000_v7_20260831T101333Z`

Fresh Legion audit: **2026-09-01 09:02:03 UTC**.

- Training job `1911789` is **COMPLETE**. The workload exited `0`, its durable
  archive is verified, and it is no longer present in `squeue`.
- Trained on: **no arrows**, dataset variant `control`, policy
  `action_visual_lora_v1`, seed `1000`, batch size `32`, LoRA rank `16`.
- Source commit: `98f9e295fe05400cbbce6d1e9cf500222327dac5`.
- Final progress: step `29,190 / 29,190` (**100%**), epoch `15`; the last
  progress line reported loss `0.409` at `2026-09-01 04:28:59 UTC`.
- Final saved checkpoint: `029190`, written at `2026-09-01 04:35:02 UTC`.
  All 15 scheduled checkpoints exist, the post-training adapter audit passed,
  and the final adapter reload smoke test passed.
- Run directory:
  `/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/runs/legion_action_visual_lora_no_arrow_s1000_v7_20260831T101333Z_candidate`.
- Training completed at `2026-09-01 04:36:10 UTC`; no fatal signature appears
  in the training log. Verified archive tree SHA-256:
  `33a84486fc8b9c5371d1075a8a74da0020b3b00a48963b1ad1cc91fe3d000885`.
- Dependent evaluation job `1911790` ran after training and completed both
  100-episode no-arrow cells. The new action-visual checkpoint scored
  **26/100 (26%)** with per-task successes `2, 8, 0, 1, 0, 0, 3, 4, 8, 0`;
  the historical action-only checkpoint scored **43/100 (43%)** with per-task
  successes `9, 9, 6, 4, 1, 1, 6, 4, 3, 0`.
- Job `1911790` nevertheless exited `1` because the policy cells did not use
  identical reset/randomization identities. The two individual `eval_info.json`
  results are complete and archived, but **their difference is not a valid
  paired comparison**. The verified evaluation archive tree SHA-256 is
  `77e52ebcf28a150b5e1e7f5f7be09a7690e9824aa2632fa9a90d0625ae9d4f02`.
- Clarification: job `1911790` was **two policy cells under no arrows**
  (new action-visual LoRA versus historical action-only LoRA), not an
  arrows-versus-no-arrows pair. The separate arrows-versus-no-arrows job was
  `1912060` at checkpoint `007784`: live arrows completed at `15/100`, while
  the no-arrow cell was cancelled before any episode. There is no final
  checkpoint `029190` live-arrow evaluation yet.
- Training has no success-rate result. Evaluation scores remain separate.

## Intermediate action-visual checkpoint evaluation — stopped after arrows

Fresh Legion audit: **2026-08-31 19:38:08 UTC**.

- SLURM job `1912060` evaluated checkpoint `007784` (epoch
  `4.0014136546`) from training job `1911789`.
- Trained on: **no arrows** with policy `action_visual_lora_v1`.
- Live-arrow scenario: **complete and final**, tasks 0–9 successes out of 10 =
  `1, 4, 1, 0, 1, 1, 4, 2, 1, 0`; total **15/100 (15%)**.
- No-arrow scenario: initialization started at `2026-08-31 19:02:48 UTC`, but
  it completed **zero episodes**. At the user's request, job `1912060` was
  cancelled at `2026-08-31 19:37:33 UTC`; SLURM reports `CANCELLED`, exit
  `0:15`. There is no no-arrow `eval_info.json` and therefore no no-arrow score.
- The job's exit handler preserved and hash-verified the completed live-arrow
  `eval_info.json`, audit files, checkpoint snapshot, provenance, and empty
  no-arrow audit placeholder under:
  `/home/hjaber/EmbodimentSemantic_archive/eval/legion_action_visual_lora_no_arrow_s1000_v7_20260831T101333Z_checkpoint_007784_arrows_then_none_eval_1912060`.
- Archive tree SHA-256:
  `6c70663ca8715771ed42608629f25e9cded07b187ece6b2b526deeeb73245b70`.
- Training job `1911789` was not cancelled and later completed successfully.
  Its dependent final evaluation job `1911790` also ran; its current status and
  results are recorded in the completed-training section above.

## Historical failed job chains — superseded by active v7 training

Historical Legion audit: **2026-08-31 08:37:13 UTC**. These are retained as
failed-attempt provenance and are not the current active training.

- **Action-visual LoRA:** training job `1911374` (`action_visual_lora_no_arrow_s1000_v2_20260830T200506Z`) is `PENDING` with `DependencyNeverSatisfied` because setup `1911373` failed while building the legacy action-only evidence bundle. Its dependent evaluation job is `1911375`, still dependency-blocked. No training runtime, checkpoint, or `eval_info.json` exists.
- **Graph-text LoRA:** training job `1911247` is `PENDING` with `DependencyNeverSatisfied` because setup `1911244` failed graph-policy inventory validation. Its dependent evaluation job is `1911248`, still dependency-blocked. No training runtime, checkpoint, or `eval_info.json` exists.
- The later graph setup retry `1911474` also failed before issuing a new training
  job. These failed chains have no per-task success rates; active v7 training is
  tracked in the current section above.

## Permanent takeover checklist

1. Read this ledger and `evaluation_results_tracker.json` before answering any
   experiment question.
2. Apply the mandatory update transaction above on every user update request.
3. Use `eval_info.json` as the authoritative final success record. Logs and
   videos are supporting evidence and progress signals.
4. Report each run as: model, trained on, evaluated with, checkpoint, task 0
   score, task 7 score, total score, and final versus partial.
5. Never use bare `control`; say frozen base or no-arrow LoRA.
6. Default final comparisons to epoch 15 (`029190`). Track another checkpoint
   only when the user explicitly requests it; checkpoint `007784` is one such
   recorded exception.
7. Update the Markdown ledger and JSON tracker in the same operation.
8. Preserve raw results, videos, logs, checkpoints, and provenance.

## Next actions, in order

1. Treat action-visual training job `1911789` and checkpoint `029190` as the
   completed final training artifact.
2. Preserve both `eval_info.json` cell results from job `1911790`, but do not
   interpret the 26% versus 43% delta as paired because reset identities differ.
3. If an arrows-versus-no-arrows result for the final checkpoint is needed,
   launch a fresh paired evaluation with identical reset identities; the prior
   arrows result is only for checkpoint `007784`.
3. Keep the failed action-visual and graph-text chains as historical provenance;
   do not reuse their failed states.
4. Verify durable HOME archives match scratch outputs; do not launch any
   additional baseline or seed unless explicitly requested.

## Current interpretation boundary

Verified: the all-arrow LoRA epoch-15 checkpoint scored 11/20 when evaluated
with all arrows; the frozen base scored 0/20 under that same all-arrow rollout
condition.

Not yet isolated: whether the gain is caused by arrow-conditioned learning,
generic additional fine-tuning, arrows at evaluation time, or an interaction.
Historical failed jobs `1911374` and `1911247` never started. Their failures are
not evidence about active action-visual job `1911789`, which is running and has
no final evaluation result yet.
