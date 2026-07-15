# Unified LIBERO/SO101 Demo

The deployed demo is the same backend-driven application used locally:

- `server.py` serves both the LIBERO and SO101 API namespaces.
- `libero/` and `so101/` contain the two full browser applications.
- `common/` contains their shared design and rendering helpers.
- `libero_demo_cache/` contains `demo_0` MuJoCo states and semantic records for
  all ten LIBERO tasks. It contains no RGB frames; LIBERO renders frames live.
- `libero_prediction_cache/` contains compact `demo_0` model predictions.
- `so101_demo_cache/` contains real `episode_0` source frames, proxy artifacts,
  metadata, and Gemini predictions for the five SO101 tasks.

The old static mini demo and GitHub Pages deployment have been removed.

## Local

Run from `vlm_benchmarking` with the configured LIBERO environment:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo --port 7860 --no-disk-cache --no-open-browser
```

For the same one-demo dataset used by Fly.io:

```powershell
C:\Users\hassa\anaconda3\envs\vla_bench_py312\python.exe -u -m demo `
  --input-dir demo/libero_demo_cache `
  --output-dir demo/libero_prediction_cache `
  --so101-config demo/so101_demo_cache/config.yaml `
  --host 127.0.0.1 --port 7860 --no-disk-cache --no-open-browser
```

## Fly.io

`Dockerfile` installs the pinned LIBERO commit and simulator dependencies.
`fly.toml` starts the unified backend on `0.0.0.0:7860`.

```powershell
fly deploy
```

The public application is <https://embodimentsemantic.fly.dev/>.
