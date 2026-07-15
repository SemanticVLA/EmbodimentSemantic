# Unified LIBERO/SO101 Demo

This directory contains the presentation layer for both datasets:

- `libero/`: the full simulator-backed LIBERO explorer.
- `so101/`: the full SO101 2D Proxy GT explorer.
- `so101_proxy_demo/`: the SO101 bbox/proxy pipeline, artifacts, CLI, and tests.
- `index.html`: the shared LIBERO/SO101 selector.
- `libero_backend.py`: LIBERO HDF5, prediction, rendering, and API logic.
- `so101_backend.py`: SO101 artifact, frame, and API logic.
- `server.py`: one local server composing both API namespaces.
- `common/`: shared browser design tokens, layout primitives, and UI helpers.
- `data/`: the generated, compressed mini dataset used by GitHub Pages.

All LIBERO/SO101 demo code now lives under this directory. No compatibility
copy remains at `vlm_benchmarking/so101_proxy_demo/` or in `vlm_bench/`.

## Static Mini Demo

The checked-in bundle contains:

- LIBERO: `demo_0` for every task and both cameras.
- SO101: `episode_0` for every task and both cameras.
- Up to 12 uniformly sampled frames per sequence, stored as WebP.
- Compact bbox and ground-truth relation metadata in `libero.json` and
  `so101.json`.

Viewing this version does not require LIBERO, MuJoCo, HDF5, or the SO101 source
dataset. Run from `vlm_benchmarking` with any Python installation:

```powershell
C:\Users\hassa\anaconda3\python.exe -u -m demo.static_server --port 7860 --no-open-browser
```

Open `http://127.0.0.1:7860/`.

## Full Local Explorer

Run from `vlm_benchmarking`:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo --port 7860 --no-open-browser
```

The selector embeds the full local explorers when this backend is running.
LIBERO uses `/api/*` and renders frames through the LIBERO simulator instead
of the compressed static bundle. SO101 uses `/so101/api/*` and reads the local
SO101 artifacts/videos. The static mini dataset is only the GitHub Pages
fallback when no Python backend is available.

## GitHub Pages

GitHub Pages cannot execute Python, MuJoCo, or read local HDF5 files. The hosted
selector therefore reads the checked-in mini dataset while preserving the
dataset, task, camera, demo/episode, frame, bbox, arrow, and triplet controls.

Regenerate those samples after changing graphs or bboxes:

```powershell
cd C:\Users\hassa\OneDrive\Desktop\EmbodimentSemantic\vlm_benchmarking
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo.so101_proxy_demo detect --sampled --camera wrist --episode episode_0
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo.so101_proxy_demo generate-proxy
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo.build_static_bundle
```

Use `--sample-count 0` only when a full-frame export is intentionally required;
the default cap keeps the repository and Pages download small. SO101 wrist
sequences are exported only when `wrist.jsonl` covers every sampled frame in
that episode.

The repository-root `index.html` redirects to this directory. The workflow at
`.github/workflows/pages.yml` publishes only that redirect and this `demo/`
tree. In repository **Settings > Pages**, select **GitHub Actions** as the source
once; pushes to `main` then deploy the hosted demo automatically.

## Tests

Run from the repository root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
C:\Users\hassa\anaconda3\python.exe -m pytest vlm_benchmarking/tests vlm_benchmarking/demo/so101_proxy_demo/tests -q
```
