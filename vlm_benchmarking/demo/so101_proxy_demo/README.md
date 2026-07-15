# EmbodimentSemantic SO101 2D Proxy GT Sub-Demo

Last updated: 2026-07-15

This folder is the canonical record for the isolated real-world SO101 scene-graph work. Read this file before changing the pipeline. No context from the LIBERO demo or previous conversations should be required.

## Objective

Build a real-world browser demo and evaluation track that approximates the LIBERO scene-graph rules using signals available in the SO101 dataset:

1. 2D object bboxes from the fixed agent view and moving wrist camera.
2. Synchronized robot metadata such as gripper state and end-effector motion.
3. A deterministic bbox-midpoint relation rule with task-aware support and SO101 metadata gates.

The resulting labels are called **2D Proxy GT**. They are not physical 3D ground truth.

## Scientific Non-Claims

This project does not claim that:

- Bboxes establish metric object `x/y/z`.
- Projected overlap proves physical contact.
- Gripper state proves successful grasping without visual evidence.
- Wrist-image geometry defines a stable world frame.
- Pure bbox geometry eliminates the simulator-to-real domain gap.
- Automatically produced SO101 graphs are equivalent to simulator truth.

The proxy is useful because its construction is deterministic, versioned, auditable, and validated against the true LIBERO graphs wherever corresponding 2D evidence exists.

## Isolation Boundary

SO101 detection, metadata, graph, validation, CLI, tests, and artifacts live
under `vlm_benchmarking/demo/so101_proxy_demo/`. Its HTTP backend and frontend
are sibling components in `vlm_benchmarking/demo/`, the repository's only
`demo` directory.

The package reads these existing locations:

- `vlm_benchmarking/data/SO1001_dataset/`
- `vlm_benchmarking/data/SO1001_dataset/gemini-generated graphs/`

It must never write to those directories, the existing `output/` directory, existing HDF5 files, or the LIBERO demo. Every generated file goes below:

```text
demo/so101_proxy_demo/artifacts/
```

`ArtifactStore` resolves every output path and rejects traversal outside that root.

## Dataset Inventory

The current audit finds:

- 5 tasks.
- 257 usable episode metadata records.
- 120,299 usable episode-local frames per camera.
- 4,126 sampled 1 Hz agent-view JPEGs.
- 4,126 sampled 1 Hz wrist JPEGs.
- 8,252 sampled JPEGs total, matching the Gemini inference cadence.
- 30 FPS synchronized videos and robot state.
- Missing episode metadata indices `12`, `13`, `14`, and `15` in the right-cookie task.

The four missing records are reported and excluded. They are never silently reconstructed.

The `agent_view_depth` stream is intentionally ignored.

## Evidence Lanes

The implementation keeps each evidence source separate. Reports can measure each lane independently.

### 1. BBox Geometry

This lane is deterministic and untrained.

For `A = [x1,y1,x2,y2]` in an image of width `W` and height `H`:

```text
uA = (x1 + x2) / (2W)
vA = (y1 + y2) / (2H)

du = uA - uB
dv = vA - vB
```

The direct axis rule is:

```text
if abs(dv) >= abs(du):
    dv > 0  -> A is_in_front_of B
    dv < 0  -> A is_behind B
else:
    du > 0  -> A is_right_of B
    du < 0  -> A is_left_of B
```

This follows the existing canonical convention: image bottom is front and image top is behind.

The baseline support statistic matches the LIBERO notebook:

```text
IoMin = intersection(box_bowl, box_support)
        / min(area(box_bowl), area(box_support))
```

`IoMin >= 0.8` proposes `black_bowl is_on_top_of support`. The inverse is generated automatically. Only bowl pairs can become support pairs, matching the effective notebook behavior.
For real SO101 boxes, GT also accepts task-allowed support when IoMin is at least `0.7` and the normalized bbox midpoint distance is at most `0.08`. This catches visually centered bowl-on-cookie/stove cases where detector boxes make the strict `0.8` cutoff too brittle.

This lane also owns bbox visibility, IoU, containment ratios, area ratios, edge gaps, five-frame center smoothing, three-frame relation persistence, direction, and inverse generation.
Each directed triplet also records pure geometric evidence from the subject bbox midpoint to the object bbox midpoint: pixel centers, normalized arrow `dx/dy`, dominant image axis, axis margin, IoU, IoMin, and containment ratios. The browser's **GT** view displays those arrows directly.

#### Canonical fixed-camera bbox generator

The canonical agent-view generator is a clean source-image pipeline. It does **not** read an older bbox JSONL and it does not apply demo-only overrides. Its only inputs are sampled JPEGs, source video, metadata, task text, configuration, and versioned DINO proposal caches.

Reproducibility is fixed by:

- Model `IDEA-Research/grounding-dino-tiny` at revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`.
- Explicit fast processor mode.
- Versioned static-anchor and bowl-candidate records in `agent_bbox_candidates.sqlite3`.
- Deterministic endpoint, temporal-path, and optical-flow rules.
- Atomic export to `artifacts/bboxes/agent_view.jsonl`.

The model is Tiny because generation currently runs on CPU and covers 257 episodes. In the verified failures, Tiny already returned the correct bowl proposal; selection, not proposal recall, was the problem. The model remains configurable so Base can be evaluated separately without silently changing this frozen run.

The rules are:

1. Run a full DINO prompt on the first sampled frame of each episode.
2. Keep only `red_drawer`, `black_stove`, `cookie`, and `white_plate` as static anchors and propagate them unchanged.
3. Recover a missing static class with its class-specific prompt on the last, middle, then quarter episode frame.
4. Use metadata only to identify stable pre-grasp, transport, release, and settled frame ranges.
5. Run a bowl-only prompt at both endpoints and every sampled transport frame. Use threshold `0.05`, retain unlabeled bowl-only proposals, and keep area fractions from `0.002` through `0.04`.
6. Select the source with the task's drawer or left/right-of-stove prior plus first-to-last image change. Select the destination with the task's cookie/stove support prior plus change.
7. Select transport proposals jointly with a temporal path cost using center motion, endpoint area, DINO confidence, and image change. A dataset-wide area prior is only a fallback.
8. If the selected DINO box is missing or clipped against an image boundary, compute the complete forward/backward Lucas-Kanade track in one video decode and substitute the flow box when available.
9. Propagate the source before transport and the destination from the sampled release frame onward.
10. Generate Proxy GT from `agent_view.jsonl`; centered five-frame smoothing is used offline for graph geometry, while the browser draws unsmoothed synchronized boxes.

If metadata is unreliable, bowl proposals are generated for every sampled frame. Metadata controls when the bowl may move; it never chooses horizontal relation direction.

#### What was tried and rejected

- **Highest-score DINO bowl:** often selected a black cylinder, stove, or robot region instead of the bowl.
- **High threshold plus sparse stride:** accurate proposals existed below threshold, sometimes with an empty text label, so transport frames were missed.
- **Old detection plus manual patch JSON:** made selected demos look correct but was not reproducible and has been removed from both code and artifacts.
- **DINO-only waiting for a later anchor:** delayed graph changes until the bowl settled when intermediate detections were absent.
- **Optical flow for every transport box:** worked in clipped scenes but could drift when a precise DINO proposal was already available.
- **Trailing five-frame median:** introduced roughly two sampled frames of visual delay. Smoothing is now centered and offline.

#### What worked

- Detect static objects once because `agent_view` is fixed.
- Preserve all plausible low-threshold bowl-only proposals rather than trusting the top score.
- Use task support priors and image change to choose source and destination.
- Choose intermediate DINO proposals as one temporal sequence.
- Use one-pass optical flow only for missing or boundary-clipped selected boxes.
- Keep raw proposal caches separate from deterministic selection so rule changes do not rerun DINO.

Clean source-image validation currently covers:

- `place-the-black-bowl-from-on-top-of-the-drawer-to-the-stove / episode_12`
- `move-the-black-bowl-from-the-top-of-the-drawer-to-on-top-of-the-cookie-at-the-le / episode_16`
- `place-the-black-bowl-from-the-left-to-the-top-of-the-stove / episode_10`

These 49 sampled frames reproduce the visually approved source, transport, release, and destination boxes. The current canonical artifact now covers all 4,126 agent-view sampled frames; `artifacts/reports/agent_bbox_report.json` reports `complete_agent_coverage: true`.

The dedicated wrist generator runs Grounding DINO once per 1 Hz wrist sample,
caches raw candidates in `wrist_bbox_candidates.sqlite3`, and exports every
selected frame to `artifacts/bboxes/wrist.jsonl`, including empty frames. It
uses direct visual confidence, bracketed bidirectional optical flow, held-bowl
visual anchors, release cutoff, and rapid-rotation rejection. Metadata selects
recovery windows but never marks an object visible.

### 2. SO101 Metadata Gates

This lane is deterministic and untrained. It gates support confidence but never chooses left/right versus front/behind.

Available frame-level metadata:

- Measured `observation.state`: `ee.x`, `ee.y`, `ee.z`, `ee.wx`, `ee.wy`, `ee.wz`, `gripper_pos`.
- Commanded `action` with the same seven dimensions.
- Episode-local timestamp and frame index.
- Exact task text.
- Synchronized MP4 start/end timestamps.

Measured dataset behavior:

- Per-episode closed gripper centers are approximately `2`.
- Per-episode open centers are approximately `17-20`.
- The median episode has four stable open/close transitions.
- Commanded gripper transitions lead measured state by approximately 3 video frames, or 100 ms.
- Relative end-effector lift spans enough range to detect transport without camera calibration.

The phase algorithm:

1. Median-smooth measured and commanded gripper values over five frames.
2. Fit two gripper clusters independently per episode.
3. Debounce transitions with a five-frame minimum dwell.
4. Use commanded action as the candidate transition and measured state as confirmation.
5. Select an open-to-closed transition followed by at least 2 cm of relative `z` lift as grasp.
6. Select the next closed-to-open transition as release.
7. Suppress all bowl support proposals while held or lifted.
8. Switch immediately to support at release when bbox support evidence is present; otherwise retain normal relation persistence.

Metadata is unreliable when cluster separation is weak, transitions are outside the configured range, or grasp/release cannot be found. Unreliable episodes fall back to bbox rules and carry an explicit `metadata_unreliable` gate.

End-effector translation is used only to check bowl-track motion coherence. No robot-to-image transform is fitted. Rotation-change magnitude only lowers wrist tracking confidence during rapid camera movement.

Task text restricts allowed bowl supports:

- Drawer-to-cookie: `red_drawer`, `cookie`.
- Drawer-to-stove: `red_drawer`, `black_stove`.
- Left/right-to-stove: `black_stove` only.
- `white_plate` is not a valid bowl support in these five tasks.
- No current task produces `inside/contains`.

## Fusion Precedence

For each visible unordered object pair:

1. Remove pairs without visual visibility.
2. Apply metadata held/lifted support suppression when metadata is reliable.
3. Apply deterministic bbox support for task-allowed bowl/support pairs.
4. Apply the deterministic image-axis rule from bbox midpoint deltas.
5. Generate the inverse relation.

The wrist graph is never independently inferred from wrist geometry. It is the synchronized agent graph filtered to objects visible in the wrist frame.

Every triplet records its source:

- `bbox_axis`
- `bbox_support`

It also records confidence and active metadata gates.

## GT Output

`generate-proxy` creates one merged GT output:

1. `gt`: bbox midpoint geometry plus SO101 metadata gates when metadata is reliable.

When metadata is unavailable or unreliable, GT falls back to pure bbox geometry instead of inventing a grasp/release interval.

## Folder and Artifact Layout

```text
demo/so101_proxy_demo/
|-- README.md
|-- __main__.py
|-- cli.py
|-- config/default.yaml
|-- proxy/
|-- vision/
|-- tests/
`-- artifacts/
    |-- index/
    |-- metadata/
    |-- bboxes/
    |-- models/
    |-- proxy_graphs/
    |-- reports/
    |-- audit/
    `-- cache/

vlm_benchmarking/demo/
|-- index.html
|-- server.py
|-- libero_backend.py
|-- so101_backend.py
|-- build_static_bundle.py
|-- common/
|-- so101_proxy_demo/
|-- data/
|-- libero/
`-- so101/
```

Important artifacts:

```text
artifacts/index/episodes.jsonl
artifacts/index/sampled_frames.jsonl
artifacts/metadata/episode_signals.jsonl
artifacts/metadata/frame_signals.jsonl
artifacts/bboxes/imported.jsonl
artifacts/bboxes/agent_view.jsonl
artifacts/bboxes/wrist.jsonl
artifacts/cache/agent_bbox_candidates.sqlite3
artifacts/cache/wrist_bbox_candidates.sqlite3
artifacts/proxy_graphs/gt/<camera>.jsonl
artifacts/reports/agent_bbox_report.json
artifacts/reports/*.json
```

Artifacts are intentionally ignored by Git. The README, source code, tests, and configuration are tracked.

## Installation

Run from `vlm_benchmarking/`.

Core pipeline:

```powershell
python -m pip install -r demo/so101_proxy_demo/requirements.txt
```

Optional Grounding DINO detector:

```powershell
python -m pip install -r demo/so101_proxy_demo/requirements-vision.txt
```

On the current machine, run the detector through the compatible VLA environment:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -m demo.so101_proxy_demo detect --sampled
```

`detect --sampled` defaults to the canonical `agent_view` rules above. It uses `artifacts/cache/agent_bbox_candidates.sqlite3` as a transactional, versioned resume cache and atomically exports `agent_view.jsonl`. Interrupting DINO does not discard completed batches; rerun the identical command.

Console progress is split into four labeled phases with counts and percentages: planning, static anchors/recovery, bowl proposals, and canonical track construction. It also reports how many records were already cached.

Target one episode without changing the rules:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -m demo.so101_proxy_demo detect --sampled --task place-the-black-bowl-from-on-top-of-the-drawer-to-the-stove --episode episode_12
```

Run a limited number of episodes for a smoke test:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -m demo.so101_proxy_demo detect --sampled --limit 1
```

The base Anaconda environment currently has a NumPy 2.x versus SciPy/scikit-learn binary mismatch that also affects Transformers imports. The core proxy generation path is NumPy-only and does not depend on SciPy or scikit-learn.

Optional SAM 2.1 mask refinement must be installed from the official repository, then configured with `sam2_config` and `sam2_checkpoint` in `config/default.yaml`. SAM 2 is not required when importing bbox JSONL.

## Complete Execution Order

Run these commands from `vlm_benchmarking/`:

```powershell
python -m demo.so101_proxy_demo audit-data
python -m demo.so101_proxy_demo extract-metadata
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo.so101_proxy_demo detect --sampled --camera agent_view
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo.so101_proxy_demo detect --sampled --camera wrist
python -m demo.so101_proxy_demo generate-proxy
python -m demo.so101_proxy_demo validate --camera all
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo --port 7860
```

The unfiltered detector command above is the all-task/all-demo command. It is resumable and processes every one of the 257 usable episodes without skipping sampled frames. Do not run `generate-proxy` until `artifacts/reports/agent_bbox_report.json` says `complete_agent_coverage: true`.

`validate --camera all` requires both camera artifacts and checks that every
wrist graph is exactly the synchronized agent graph filtered to accepted wrist
bbox endpoints.

If bboxes are generated elsewhere:

```powershell
python -m demo.so101_proxy_demo import-bboxes C:\path\to\bboxes.jsonl
python -m demo.so101_proxy_demo generate-proxy
```

`agent_view.jsonl` takes precedence over imported boxes unless `--bboxes` is provided explicitly.

## BBox JSONL Contract

One record per camera frame:

```json
{
  "schema_version": "so101-proxy-v1",
  "task": "place-the-black-bowl-from-the-left-to-the-top-of-the-stove",
  "episode": "episode_0",
  "frame": 30,
  "timestamp": 1.0,
  "camera": "agent_view",
  "width": 640,
  "height": 480,
  "detector": "agent-bbox-rules-v7-<fingerprint>",
  "objects": {
    "black_bowl": {
      "bbox": [100, 120, 180, 200],
      "confidence": 0.95,
      "tracking_confidence": 0.91,
      "visible": true,
      "source": "agent_bowl_dino_temporal_path"
    }
  }
}
```

Accepted object aliases are canonicalized. Unknown objects and the black cylindrical distractor are discarded.

The `source` field distinguishes static DINO anchors/recovery, task-prior endpoints, DINO temporal-path boxes, optical-flow substitutions, and linear fallback.

Wrist visibility uses a stricter confidence threshold than agent view, rejects boxes covering most of the image, and removes near-identical cross-class detections. The wrist lane prefers omitting an uncertain object over hallucinating a visible graph node.

## Demo

Launch the combined LIBERO/SO101 demo from `vlm_benchmarking`:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo --port 7860
```

Open `http://127.0.0.1:7860/` and select **SO101**. The direct route is
`http://127.0.0.1:7860/so101/`.

The demo is titled **SO101 Demo** and provides:

- Agent view and wrist view when their camera-specific bbox artifacts exist.
- Merged GT mode.
- Proxy GT, Gemini prediction, and comparison presentation modes.
- Sampled-JPEG scrubbing and playback.
- Continuous source-video playback through byte-range MP4 streaming.
- Subject filtering, arrows, labels, and bbox debugging.
- Triplet provenance, confidence, gripper phase, reliability, and TP/FP/FN.

The SO101 explorer itself never derives data from LIBERO. The unified local
server also hosts the independent LIBERO explorer and initializes its simulator
only when LIBERO rendering is requested.

To stop the unified demo after launch:

```powershell
$pid7860 = (Get-NetTCPConnection -LocalPort 7860 -State Listen).OwningProcess
Stop-Process -Id $pid7860
```

## Tests and Smoke Checks

Run the isolated tests only:

```powershell
python -m pytest demo/so101_proxy_demo/tests -q
```

Known-good data smoke checks:

```powershell
python -m demo.so101_proxy_demo audit-data
python -m demo.so101_proxy_demo extract-metadata
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo --port 7860 --no-open-browser
```

Expected audit values:

```text
usable_episodes: 257
usable_episode_frames: 120299
sampled_jpegs: 8252
agent_view sampled: 4126
wrist sampled: 4126
missing metadata: [12, 13, 14, 15]
```

Expected metadata values may evolve slightly with configuration, but the current extraction finds 257 analyzed episodes, no read failures, and a median action lead of 3 frames.

## Validation and Evaluation

Validation enforces:

- One relation per visible unordered pair.
- Exactly two inverse directed triplets per pair.
- No unknown predicates.
- No relation endpoint outside the visible object set.
- Support only from `black_bowl` to a task-allowed support.
- Wrist graph as a visibility-filtered subset of the agent graph.
- Separate provenance counts for axis and support geometry.

Paper reporting should include:

- Deterministic bbox versus true LIBERO graph agreement.
- Per-relation and macro/micro triplet metrics.
- SO101 metadata reliability rate.
- Proxy confidence and source distribution.
- Gemini performance against GT.
- A 250-frame stratified human audit, with 50 independently double-checked frames.

The audit must cover initial state, approach, grasp, transport, release, final placement, diagonal pairs, and support pairs. It estimates Proxy GT noise; it does not tune the GT rules.

## Troubleshooting

### PyArrow or NumPy import errors

Use a consistent environment and reinstall compatible NumPy, PyArrow, and pandas builds. The reader uses `ParquetFile.read` directly and does not require pandas for normal operation.

### Grounding DINO is unavailable

Use `import-bboxes`. The complete metadata, fusion, validation, and demo pipeline does not require detector packages.

### SAM 2 is unavailable

Leave `sam2_config` and `sam2_checkpoint` empty. Grounding DINO boxes remain usable without mask refinement.

### Browser cannot decode AV1 source video

Sampled JPEG scrubbing still works. Use sampled playback or create an external H.264 proxy without modifying the source videos.

### Proxy GT is missing in the demo

Run bbox detection or import, then `generate-proxy` and `validate`. The demo intentionally starts without fabricated bboxes.

### Metadata is marked unreliable

Inspect `artifacts/metadata/episode_signals.jsonl`. The pipeline falls back to bbox geometry instead of inventing a grasp/release interval.

### Validation reports incomplete coverage

Compare the bbox-frame count against 4,126 frames per camera. Missing detector records cannot produce complete Proxy GT.

### Start bbox generation from an empty state

Stop the demo server first. Remove only generated SO101 bbox state and proxy graphs under this package:

```powershell
Remove-Item -LiteralPath demo\so101_proxy_demo\artifacts\bboxes\agent_view.jsonl -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath demo\so101_proxy_demo\artifacts\cache\agent_bbox_candidates.sqlite3 -Force -ErrorAction SilentlyContinue
Get-ChildItem demo\so101_proxy_demo\artifacts\proxy_graphs -Recurse -File -Filter *.jsonl | Remove-Item -Force
```

Never run a cleanup command against `data/SO1001_dataset`, `output/`, or another project artifact root.

## Status Checklist

Status as of 2026-07-14:

- [x] Isolated package and artifact write guard.
- [x] SO101 audit and sampled-frame index.
- [x] Missing metadata reporting.
- [x] Gripper/action/state phase extraction.
- [x] Task support priors.
- [x] Deterministic bbox geometry.
- [x] Optional Grounding DINO and SAM 2 adapter.
- [x] Versioned bbox importer.
- [x] Pure geometric bbox evidence and metadata-gated fusion outputs.
- [x] Wrist visibility filtering.
- [x] Proxy validation.
- [x] Standalone image/video API and browser UI.
- [x] Isolated automated tests.
- [x] Real dataset audit artifact generated locally.
- [x] Real metadata artifacts generated locally.
- [x] Removed LIBERO surrogate path from active SO101 proxy generation.
- [x] Document rejected high-score, sparse-stride, broad-flow, trailing-median, and manual-override approaches.
- [x] Remove legacy SO101 bbox output, detector cache, and demo override artifacts.
- [x] Pin DINO Tiny revision and processor mode.
- [x] Clean source-image agent generator with no legacy bbox input.
- [x] One static-object detection policy plus class-specific recovery.
- [x] Low-threshold bowl candidate cache separated from deterministic rules.
- [x] Task-prior endpoint selection and temporal DINO path.
- [x] One-pass optical flow for missing or boundary-clipped selected boxes.
- [x] Centered offline smoothing; raw synchronized browser geometry.
- [x] Clean smoke validation on drawer-to-stove, drawer-to-cookie, and left-to-stove episodes (49 frames).
- [x] Isolated test suite: 42 passing tests.
- [x] Run canonical generation across all 257 episodes / 4,126 agent frames.
- [x] Confirm `complete_agent_coverage: true` with zero failed episodes.
- [x] Regenerate merged `gt` agent-view Proxy GT artifact from `agent_view.jsonl`.
- [x] Validate 4,126 GT frames with zero invalid graph frames.
- [ ] Re-enable wrist generation only when it becomes a current priority.
- [ ] Perform the 250-frame human audit.
- [ ] Freeze paper metrics and curated demo episodes.

The immediate next step is the stratified human audit and paper-metric freeze. Do not label Gemini output as GT and do not bypass the provenance fields.
