# Canonical grasp controller internals

This package is the implementation behind the single public entrypoint
`vla_benchmarking.run_grasp_controller`. It is intentionally modular, but its
modules are not independent operator policies:

- `runner.py` owns capture provenance, candidate ordering, bounded retries, and
  the episode adapter;
- `molmopoint.py` owns the pinned MolmoPoint proposal worker;
- `grasp_candidates.py` owns RGB-D support checks and executable grasp
  geometry;
- `preshape.py` owns measured gripper opening/preshape behavior.

The Legion operator surface is the one launcher
`vla_benchmarking/legion/run_grasp_controller.sbatch`. Do not invoke these
modules directly for a new evaluation.

## Immutable default contract

The package preserves the verified `failure_opening40_retreat80` behavior from
release `b4fb87759ae3a1ea2cd518cd201a1a737bb14e80`:

1. Capture aligned agentview RGB-D and the existing calibration together.
2. Use the arrow-derived bowl support and MolmoPoint's clearance-aware rim
   proposals; do not use SAM.
3. Expand valid proposals into executable jaw positions, yaw, insertion,
   opening, approach, release, and retreat poses using RGB-D geometry.
4. Rank deterministically and execute at most four attempts. After an empty
   close, close/lift timeout, or failed retention indication, open, retreat,
   capture a fresh frame, regenerate candidates, and select a different one.
5. Apply the measured 40 mm preshape, 20 mm release-height compensation, and
   80 mm post-release retreat while preserving the existing phase and action
   limits.
6. Query the evaluator only after placement and retreat.

Model revision, prompt identity, camera/depth contract, calibration, release
identity, and candidate/retry records are written to the run manifest. Missing
or inconsistent provenance fails closed. The runner must not consume
simulator object poses or evaluator results to choose a grasp.

## Frozen evidence and rollback

The behavior-equivalent frozen release produced **87/100 (87%)** on
sealed-randomized Legion job `1920556`. The raw output archive is:

```text
/home/hjaber/EmbodimentSemantic_archive/molmo_failure_sealed100/
molmo_failure_sealed100_fa1ae83_1920556
```

Keep Git history and Legion archives as the rollback/audit record. A behavior
change requires a new experiment identity and a new release; it must not be
introduced as a hidden variant of this package.

The 87% result is evidence for the executed release above. A later
behavior-equivalent refactor must pass its own smoke/regression checks before
being described as separately evaluated.
