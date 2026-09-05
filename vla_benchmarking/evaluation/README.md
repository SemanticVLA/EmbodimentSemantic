# Shared LIBERO Spatial evaluation

This package is the single evaluation home for the canonical grasp controller,
fine-tuned VLA policies, and baseline LeRobot policies. It owns experiment
conditions, task/seed schedules, randomization, visual inputs, text context,
audits, and result aggregation. Policy packages continue to own inference and
robot actions.

## Independent evaluation factors

- Suite: `normal` or `sealed_randomized`. The direct historical backend records
  `normal` as its native value, `vanilla`, to preserve existing manifests.
- Visual input: `none`, `goal_arrow`, or `relation_arrows`.
- Text input: `none`, `scene_graph`, or `text_triplet` with an explicit format
  from `scene_graph_formats.py`.

`contracts.py` defines the stable condition and cell schedule. `registry.py`
declares which combinations each policy accepts and fails closed on unsupported
inputs.

## Two execution backends

- `run_arrow_pick_place_matrix.py` is the direct per-cell backend used by the
  canonical controller and language-free ArrowStudent. Their policy adapters
  retain their own motion/inference loops.
- `run_lerobot_eval_with_context.py` is the LeRobot backend used by base and
  LoRA VLA policies. It retains the existing processor, camera, vector-env,
  reset, and `eval_info.json` behavior.

The package does not merge these into a universal action loop. Evaluator
outputs are reporting data only and never enter candidate or action selection.

Root-level scripts such as `run_lerobot_eval_with_context.py` and
`run_arrow_pick_place_matrix.py` are compatibility wrappers for one transition
release. New imports should use `vla_benchmarking.evaluation`.
