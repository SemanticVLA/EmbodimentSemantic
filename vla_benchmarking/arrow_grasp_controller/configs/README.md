# Canonical grasp controller

This directory contains the one supported grasp-controller declaration. The
canonical release is the frozen `failure_opening40_retreat80` treatment, with
the neutral runtime identity used by `run_grasp_controller.py`. There are no
active legacy controller choices.

The checked-in declaration is
[`canonical_molmo_rgbd_grasp.json`](canonical_molmo_rgbd_grasp.json).

The supported operator entrypoints are:

```text
python -m vla_benchmarking.arrow_grasp_controller.run_grasp_controller --output-dir /absolute/output/path
sbatch vla_benchmarking/arrow_grasp_controller/legion/run_grasp_controller.sbatch  # with required release env
```

Use the entrypoint and launcher documentation as the source of truth for
arguments. Internal geometry/model modules are implementation details and are
not separate launch paths.

## Frozen behavior

The default preserves the behavior of the verified 87/100 sealed-randomized
release:

- `agentview` RGB-D capture, with aligned metric depth and the existing camera
  calibration/transform contract;
- no SAM or SAM endpoint;
- MolmoPoint-8B for multiple rim-contact proposals, using the pinned model
  revision and the clearance-aware pointing prompt;
- RGB-D geometry for contact position, jaw direction, opening, insertion,
  approach, transfer, release, and workspace/obstruction checks;
- up to four grasp attempts, with a fresh capture and regenerated candidates
  after a failed close, lift, or retention check;
- measured approximately 40 mm preshape and the 20 mm release-height /
  80 mm retreat treatment;
- the existing phase limits, action budget, retention gate, evaluator timing,
  and placement compensation.

The controller receives clean RGB, the arrow-derived task endpoints, aligned
depth, calibration, and proprioception according to the existing vision and
motion contracts. It does not use simulator object poses or evaluator output
to select a grasp. The evaluator is queried only after placement and retreat.

## Release identity and evidence

The behavior-equivalent source release is
`b4fb87759ae3a1ea2cd518cd201a1a737bb14e80`; it produced the final
sealed-randomized Legion job `1920556`:

| Task | Our |
| ---: | ---: |
| 0 | 10/10 |
| 1 | 10/10 |
| 2 | 10/10 |
| 3 | 10/10 |
| 4 | 7/10 |
| 5 | 10/10 |
| 6 | 8/10 |
| 7 | 10/10 |
| 8 | 10/10 |
| 9 | 2/10 |
| **Overall** | **87/100 (87%)** |

Raw results, manifests, provenance, and the archived release are preserved at
the Legion archive recorded in the handoff documentation:

```text
/home/hjaber/EmbodimentSemantic_archive/molmo_failure_sealed100/
molmo_failure_sealed100_fa1ae83_1920556
```

The 87% evidence belongs to that executed release; it is not a claim about a
later refactor until that refactor is independently evaluated. The commit and
archive are rollback references, not additional active
policies. Historical experiment records remain available in Git and Legion
archives for audit and reproducibility; they are not selectable runtime
defaults.

## Compatibility boundary

The old experiment labels and their launchers are retired. A stale reference
must fail clearly before environment construction rather than silently
selecting another treatment. Do not revive, rename, or mix historical
configuration hashes into the canonical release. Any future behavioral change
requires a new experiment identity and a new frozen release after evaluation.
