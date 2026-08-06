# Unified LIBERO/SO101 Demo

This directory contains two related but separate ways to use the browser app:

1. **Online cached demo**: the Docker/Fly-hosted showcase with bundled data only.
2. **Offline localhost tool**: the local workbench for rendering, inspecting, and creating scene graphs from local datasets/artifacts.

Both modes use the same browser UI and HTTP server, but they intentionally use different defaults and data roots.

## Folder Ownership

The root `demo/` layout is kept Docker-friendly: the online cache folders stay as direct children of `demo/` because the production image copies `vlm_benchmarking/demo` and `python -m demo online` resolves paths from this package root.

Online cached-demo assets:

```text
demo/
|-- online_cached_demo/        mode marker and folder map
|-- libero_demo_cache/         bundled LIBERO demo_0 states and semantics
|-- libero_frame_cache/        bundled LIBERO 1024px JPEG frame archives
|-- libero_prediction_cache/   bundled LIBERO demo_0 prediction JSONL
`-- so101_demo_cache/          bundled SO101 episode_0 frames, proxy GT, metadata, predictions
```

Offline localhost-tool workspace:

```text
demo/
|-- offline_localhost_tool/    mode marker and folder map
`-- so101_proxy_demo/          SO101 proxy source, tests, config, and writable artifacts

vlm_benchmarking/
|-- data/libero_spatial_v5/    local LIBERO source HDF5s
|-- data/SO1001_dataset/       local SO101 source dataset
|-- output/                    local LIBERO prediction outputs
`-- .cache/scene_graph_demo/   local render/semantic cache
```

Do not move the online cache directories under `online_cached_demo/` without also updating `LAUNCH_MODES` in `server.py`, `Dockerfile`, and the deployment smoke checks.

## Commands

Run commands from `vlm_benchmarking/`.

### Online Cached Demo

Use this for Docker/Fly or to reproduce the public hosted app locally:

```powershell
python -u -m demo online
```

This mode is a frozen, cache-only showcase:

- LIBERO reads `demo/libero_demo_cache/`, which contains only `demo_0` states and semantic records for the ten bundled tasks.
- LIBERO images are served from `demo/libero_frame_cache/`; the simulator is not required and no new LIBERO frames are rendered.
- LIBERO predictions are read from `demo/libero_prediction_cache/`.
- SO101 reads `demo/so101_demo_cache/config.yaml`, which points at the reduced `demo/so101_demo_cache/` bundle.
- SO101 currently stays reduced in the online version: five tasks, `episode_0`, sampled frames, proxy artifacts, metadata, and cached Gemini predictions.
- SO101 graph editing, review status, and CSV export are disabled; edit/review/export API calls return `403`.
- The server binds `0.0.0.0:7860`, disables browser auto-open, and disables runtime disk cache.

The online mode is for viewing and comparing cached examples. It is not the scene-graph generation workflow.

### Offline Localhost Tool

Use this when working locally with the full datasets, simulator rendering, writable caches, and SO101 proxy artifacts:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo offline --port 7860
```

This mode is the localhost scene-graph tool:

- LIBERO reads `data/libero_spatial_v5` and renders frames live through LIBERO/MuJoCo when needed.
- LIBERO rendered frames and decoded HDF5 semantics can be cached under `.cache/scene_graph_demo`.
- LIBERO predictions default to `output/`.
- SO101 reads `demo/so101_proxy_demo/config/default.yaml`, which points at the local SO101 dataset, local Gemini predictions, and writable `demo/so101_proxy_demo/artifacts/`.
- SO101 proxy artifacts can be created or refreshed with `python -m demo.so101_proxy_demo ...`.
- SO101 manual frame edits are saved as `output/so101_graph_edits.jsonl`.
- SO101 frame review status is saved separately as `output/so101_review_status.jsonl`.
- SO101 CSV export writes one timestamped run folder under `output/annotated_graphs/`.

The legacy command `python -m demo` still defaults to offline mode for compatibility, but new usage should say `online` or `offline` explicitly.

### Offline SO101 Graph Editor

In offline mode, the SO101 right-side panel has two pages: **Annotation** and **Eval**. Annotation is the localhost relation-label editing tool. Eval keeps metrics, metadata, wrist coverage, and triplet inspection. The browser does not regenerate proxy artifacts; it stores manual relation-label edits as an overlay on top of the generated proxy graph.

Annotation workflow:

1. Start `python -u -m demo offline --port 7860`.
2. Open SO101 and use `agent_view` as the primary graph-authoring lane. `wrist` remains available but is currently partial until full wrist artifacts are generated.
3. Offline SO101 opens on **Annotation** by default. Online SO101 opens on **Eval** and keeps annotation disabled.
4. For each frame, edit the object-pair rows in **Annotation**. The subject/object endpoints stay fixed from bboxes; only the predicate text shown on the arrow changes.
5. Use the **Worklist** filters to move through generated/edited and valid/invalid/stale frames.
6. Use `Save`, `Next`, `Reset`, or `Export CSVs`. Unsaved edits block task, camera, episode, frame, playback, and export actions.
7. Use **Eval** to inspect metrics, metadata, wrist coverage, and triplet rows.
8. Use `Export CSVs` after saving all edits. The export folder contains one CSV per camera view with an `edited` column.

Actions:

- `Save`: persists the current frame graph edit to `output/so101_graph_edits.jsonl`.
- `Next`: advances only when the current frame has no unsaved edits.
- `Reset`: removes the manual edit for the current frame and returns to generated proxy GT.
- `Export CSVs`: exports all available SO101 generated graphs with saved manual edits applied.

Shortcuts:

- `Ctrl+S`: save.
- `N`: next frame when the current frame is clean.
- Left/right arrow keys: previous/next frame when the current frame is clean.

CSV columns are:

```text
task,episode,frame,timestamp,camera,mode,subject,relation,object,edited,original_relation
```

Export layout:

```text
output/annotated_graphs/
`-- <timestamp>/
    |-- agent_view.csv
    `-- wrist.csv
```

`edited` is `yes` when that relation label was manually changed from the generated proxy graph and `no` for generated rows. `original_relation` is filled only for `yes` rows, so you can see what the generated proxy relation was before the manual label change. Export still validates saved edits before writing and fails if an edit is stale or invalid.

The edit log is JSONL at `output/so101_graph_edits.jsonl`. Each saved record uses:

```text
schema_version, task, episode, frame, camera, mode, base_graph_hash, edit_revision, updated_at, relations, validation_status, validation_errors
```

`relations` contains directed triplet rows. The editor shows one object pair per row, and the backend writes both directed triplets with the required inverse predicate.

The review log is JSONL at `output/so101_review_status.jsonl`. Each saved record uses:

```text
schema_version, task, episode, frame, camera, mode, base_graph_hash, review_status, reviewed_at, reviewer, note
```

`review_status` is `reviewed` or `needs_attention`. A missing record means the frame is unreviewed. The simplified browser UI no longer writes review records, but the backend keeps these endpoints for compatibility. Export blocks if a saved edit or saved review status is stale against the current generated `base_graph_hash`.

SO101 API surfaces used by the browser:

- `GET /so101/api/worklist`: filter by `task`, `episode`, `camera`, `mode`, `edit_status`, and `validation_status` from the browser. The backend still accepts `review_status` for compatibility.
- `GET /so101/api/frame`: returns graph edit metadata, review metadata, `base_graph_hash`, validation errors, and stale flags.
- `POST /so101/api/graph-edits`: saves pair-based relation-label overlays and requires `base_graph_hash`.
- `POST /so101/api/graph-edits/reset`: removes the edit overlay for one frame.
- `GET /so101/api/review-status`: returns one frame's review overlay in offline mode.
- `POST /so101/api/review-status`: saves one frame's review overlay and requires `base_graph_hash`.
- `POST /so101/api/review-status/reset`: removes one frame's review overlay.
- `GET /so101/api/pipeline-status`: reports artifact paths, source artifact availability, coverage, progress, and stale review records.
- `POST /so101/api/export-csv`: writes the clean per-view CSV export.

## SO101 Proxy Pipeline

The SO101 browser view does not create proxy GT by itself. Generate or update SO101 artifacts through the isolated pipeline:

```powershell
python -m demo.so101_proxy_demo audit-data
python -m demo.so101_proxy_demo extract-metadata
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo.so101_proxy_demo detect --sampled --camera agent_view
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo.so101_proxy_demo detect --sampled --camera wrist
python -m demo.so101_proxy_demo generate-proxy
python -m demo.so101_proxy_demo validate --camera all
```

Use the same commands to close the current wrist gap. In this workspace `agent_view` is complete locally, while `wrist` is partial; the offline UI and `/so101/api/pipeline-status` report the gap so reviewers do not mistake partial wrist data for full coverage.

Current local state checked from this workspace:

- Online SO101 cache: 5 tasks, 170 sampled frame records, 85 GT frames for `agent_view`, and 85 GT frames for `wrist`.
- Offline SO101 artifacts: 257 episode records, 8,252 sampled frame records, 4,126 GT frames for `agent_view`, and 85 GT frames for `wrist`.

So SO101 is intentionally reduced online. Offline has full `agent_view` proxy coverage locally, while `wrist` is currently only partially generated.

## Browser App

- `server.py` serves both the LIBERO and SO101 API namespaces.
- `index.html`, `portal.css`, and `portal.js` provide the top-level mode-aware portal.
- `libero/` and `so101/` contain the two browser applications.
- `common/` contains shared rendering and UI helpers.

The health endpoint reports the active `demo_mode`, `demo_mode_label`, and `demo_dataset_scope`; the portal displays this so users can tell whether they are looking at the cached hosted demo or the localhost tool.

## Cache Builders

- `build_libero_demo_cache.py`: creates the reduced LIBERO `demo_0` HDF5 cache without RGB frames.
- `cache_libero_frames.py`: renders the bundled LIBERO JPEG frame archives used by `online`.
- `build_libero_prediction_cache.py`: compacts LIBERO `demo_0` prediction JSONL files.
- `build_so101_demo_cache.py`: creates the reduced SO101 `episode_0` cache used by `online`.
- `deployment_smoke.py`: verifies deploy/runtime assets.

## Fly.io

The production Docker target runs:

```powershell
python -u -m demo online
```

Deploy with:

```powershell
fly deploy
```

The public application is <https://embodimentsemantic.fly.dev/>.
