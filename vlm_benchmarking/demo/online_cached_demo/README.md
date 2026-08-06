# Online Cached Demo

This folder is a mode marker and folder map. The online assets intentionally
remain as sibling directories at `demo/` root so the Docker image can copy and
serve them without special path handling.

Online mode command:

```powershell
python -u -m demo online
```

Online-owned folders:

- `../libero_demo_cache/`
- `../libero_frame_cache/`
- `../libero_prediction_cache/`
- `../so101_demo_cache/`

These are read-only bundled assets for the hosted showcase. Do not use this
mode as the scene-graph generation or annotation workflow. SO101 is intentionally
reduced to cached `episode_0` coverage for review. Graph editing, review status,
and CSV export are disabled in this mode; edit, review, and export API endpoints
return `403`.
