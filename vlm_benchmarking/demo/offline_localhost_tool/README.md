# Offline Localhost Tool

This folder is a mode marker and folder map. The offline tool uses local
datasets and writable artifacts rather than the reduced online cache.

Offline mode command:

```powershell
python -u -m demo offline
```

Offline-owned workspace:

- `../so101_proxy_demo/`
- `../../data/libero_spatial_v5/`
- `../../data/SO1001_dataset/`
- `../../output/`
- `../../.cache/scene_graph_demo/`

SO101 proxy generation is run through `python -m demo.so101_proxy_demo ...`;
the browser reads the generated artifacts but does not create them by itself.

SO101 manual frame edits are saved as `../../output/so101_graph_edits.jsonl`.
SO101 review marks from older workflows are saved separately as
`../../output/so101_review_status.jsonl`. The simplified browser annotation
page no longer writes review records; it stores relation-label overlays only.
Generated proxy artifacts remain immutable.

The offline SO101 workflow is:

1. Use the worklist filters to find generated, edited, invalid, or stale frames.
2. Edit only the relation predicate for each visible object pair; bbox
   endpoints and arrows stay generated from artifacts.
3. Use the four Annotation buttons: `Save`, `Next`, `Reset`, and `Export CSVs`.
4. Export CSVs for paper artifacts after all saved overlays validate.

CSV exports are written under a timestamped folder in
`../../output/annotated_graphs/`, with one file per camera view:
`agent_view.csv` and `wrist.csv`. Each row has an `edited` column with `yes`
or `no` so generated and manually edited graph rows are easy to separate.
