# Active campaign: v9d + MolmoPoint, without SAM

The active `run_v9d_molmo_campaign.py` uses v9d's arrow-seeded RGB-D region
as observed support for MolmoPoint and grasp geometry. It does not require,
load, or call SAM or a Triton endpoint. The original SAM implementation below
is retained as historical experimental source.

Submit `legion/v9d_molmo_campaign.sbatch` with an isolated `REPO_ROOT`, its
exact `CANARY_EXPECTED_COMMIT`, and a unique `CANARY_LABEL`. One A40 allocation
runs the following arms sequentially using one persistent MolmoPoint model:

- Dense agentview, downward-approach prompt (first live cell).
- RGB-D geometry-only agentview (no Molmo inference).
- Local Molmo regions, agentview.
- Dense agentview, contact-location prompt.
- Dense agentview, finger-clearance prompt.
- Dense wrist, downward-approach prompt.

Each screen uses tasks 4, 6, 9; seeds 1000 and 1001; vanilla and
sealed_randomized (12 planned cells per arm). The best two executable arms
extend to 60 cells each, preserving completed prefix cells and failures.
Ranking uses successes/planned, retained lifts, actions, then perception time.
Repeated operational failure on two distinct cells pauses that arm; coordinate,
evaluator, or action-contract violations stop the campaign. Results and failures
are preserved in `campaign.json` plus per-cell matrix and grasp audits.

This is an exploratory comparison of the complete grasp pipeline. RGB-D region
edges are depth-derived support boundaries, not guaranteed semantic bowl rims.
The geometry control shares the treatment motion, observation hover, and retry
limits. Frozen v9d remains at `fd24a4c5cf8da4991013ab18b15704523ad0836b`;
historical baseline scores come from different code revisions. No result changes
the default automatically. The original checkout's ledger edits are untouched.

## Historical project-local SAM3.1 runtime

This directory owns the SAM3 image-segmentation dependency for the Molmo/SAM3
canary. It does **not** import, proxy, or share the Omnis SAM3 implementation.
The runtime requires a local checkout of the pinned upstream repository at
`96914d2425f90a64f45ca977c2b5165418099543`; the expected checkpoint digest is in
[`sam3_source.lock.json`](sam3_source.lock.json).

The checkout and weights are intentionally not committed. On a GPU worker,
the canary launcher materializes a project-local checkout (or the immutable
GitHub archive for this historical revision) and writes a
`.canary_source_commit` marker. Pass that path as
`Sam3RuntimeConfig.sam3_source_dir`, together with the authorized checkpoint
path. `Sam3Runtime` verifies the pinned source marker/git revision and
checkpoint SHA-256 before importing or constructing the model. Any path
containing an `omnis` directory is rejected.

The public boundary is:

```python
request = Sam3Request(rgb=rgb_uint8_hwc, prompt="bowl")
result = Sam3Runtime(config).predict(request)
for detection in result.detections:
    # detection.mask: bool[height, width]
    # detection.box_xyxy: original-image pixel coordinates
    # detection.score: [0, 1]
    pass
```

Imports are lazy: CPU-only tests can exercise image/output validation without
PyTorch, CUDA, SAM3, or checkpoint files. The model is loaded once per runtime
and reused for subsequent frames.

MolmoPoint prompt wording is an explicit experiment factor. The default
`rim_downward_approach` prompt requests multiple distinct outer-rim contact
centres with visible depth support and finger clearance; `rim_contact` and
`rim_clearance` are selectable only as new, separately identified experiments.
Geometry still derives yaw, aperture, orientation, and insertion from the
SAM3 mask and RGB-D frame, so a point is never treated as an executable pose
by itself.

## Legion canary launch

The checked-in launcher is
`vla_benchmarking/legion/molmo_sam3_canary.sbatch`. Submit it from this exact,
clean release checkout with `CANARY_EXPECTED_COMMIT` set to the release SHA.
The authorized SAM3.1 checkpoint must be staged at `SAM3_CHECKPOINT` (or the
project-local default) and match the pinned SHA-256; the launcher fails closed
when it is missing or under an Omnis path. Set `CANARY_SETUP_SOURCE=1` on the
first job to materialize the pinned SAM3 source into the project-owned runtime.

Run the 12-cell prefix with `CANARY_PHASE=prefix`. To extend the same results
to 60 cells, submit `CANARY_PHASE=full60` with `OUTPUT_ROOT` pointing at the
prefix job's existing `results` directory; terminal prefix outcomes are kept
and only the new cells are executed. Each prompt ID and camera variant is a
separate experiment identity.
