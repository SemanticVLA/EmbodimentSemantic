# Qwen3-VL-30B frame-level scene-graph predictions

This directory preserves the Qwen3-VL-30B outputs from the `dev` branch at
commit `ada88b06b9b166b531d949b028b41cc1176f6512` (`Qwen VL 30B benchmarking`).

It contains 40 files for ten LIBERO black-bowl placement tasks:

- `agentview/csv`: 10 normalized relation tables
- `agentview/json`: 10 raw JSONL model-output files
- `eye_in_hand/csv`: 10 normalized relation tables
- `eye_in_hand/json`: 10 raw JSONL model-output files

The CSV schema is:

```text
task,demo,frame,camera,objectA,relation,objectB
```

The JSONL records additionally preserve the model name, input hash, raw
relation response, and latency. These files are predictions and provenance,
not the source images or a complete rerunnable inference configuration.
