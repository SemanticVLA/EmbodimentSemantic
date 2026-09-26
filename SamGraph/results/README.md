# SamGraph result records

This directory intentionally contains only compact, portable evidence suitable for a source repository:

- `libero_agentview_500.json`: aggregate metrics for 500 LIBERO task-episodes.
- `libero_wrist_10ep.json`: aggregate metrics for 10 episodes per task across the 10 LIBERO-Spatial tasks.
- `so101_agentview_50.json`: coverage summary for the selected 50 SO101 episodes; no accuracy metrics are claimed because verified triplet ground truth is unavailable.
- `libero_agentview_example.png`: one qualitative graph overlay.

The JSON records omit local paths, usernames, cluster job identifiers, terminal transcripts, raw masks, and source archives. Full run artifacts should be distributed through a versioned artifact host rather than committed to the source repository. Each hosted bundle should publish its URL and SHA-256 checksum alongside the paper release.

The LIBERO `agentview` record preserves the evaluation's original per-frame bowl-identity swap policy. Results produced under a different identity policy are not directly comparable.
