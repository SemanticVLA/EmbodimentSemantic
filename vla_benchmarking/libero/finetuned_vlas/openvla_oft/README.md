# OpenVLA-OFT no-arrow integration

This package defines the import-safe contract for the matched OpenVLA-OFT
baseline. It does not launch training. The native runtime is the official
OpenVLA-OFT repository in a dedicated Python 3.10 environment, initialized
from the already fine-tuned `openvla/openvla-7b-finetuned-libero-spatial`
artifact after resolving an immutable revision.

The input contract is two 256×256 cameras, 8-D proprioception, and language;
the output is continuous 7-D action chunks with the native 8-step horizon.
Training uses RLDS serialization, L1 action regression, LoRA rank 32, BF16,
microbatch 1 with accumulation 32, effective batch 32, 15 epochs / 29,180
optimizer updates, seed 1000, learning rate `5e-4`, and no-op filtering recorded
in the lineage manifest. Training augmentation and evaluation center crop are
separate, explicit stages.

Use `rlds_builder.write_rlds_source_jsonl` followed by
`build_tfds_dataset` to materialize a real episode-structured TFDS/RLDS
dataset named `libero_spatial_no_noops`. That is an existing dataset/config
and standardization transform in the pinned OpenVLA-OFT fork, so its
`RLDSDataset` can consume the generated data and write statistics under the
same key. The training launcher imports `register_tfds_builder` in-process
before invoking the exact upstream `vla-scripts/finetune.py` command; this is
required because TFDS builder registration is process-local. A flat TFRecord
stream is not a supported substitute.

The source writer applies the pinned upstream `zero_action_filter`
automatically: normalized action dimensions 0–5 must exceed `1e-5`; the
7th gripper dimension remains part of the action contract but does not change
the pinned no-op decision. It reindexes retained steps and persists
`<source>.noop_filter_receipt.json` with before/after counts plus source,
filtered-output, and output-file SHA-256 values. `build_tfds_dataset` fails
closed when that receipt is absent, stale, or does not prove the exact filter;
`build_manifest` likewise requires every frame marker and the receipt mapping
(or receipt path), embedding the verified counts/hashes. The dataset name
alone never claims filtering was performed.

Production preflight/execute commands must receive an explicit local checkout
through `--upstream-repo` and `--upstream-commit`. The checkout HEAD must be
`e4287e94541f459edc4feabc4e181f537cd569a8`, and command generation resolves
the finetune/eval scripts to absolute paths. The A40 launchers require
`OPENVLA_OFT_REPO` and `OPENVLA_OFT_COMMIT` and validate both before launch.

For shared evaluation-plan v2, pass the optional `provenance=` argument to
`OpenVLAOFTAdapter` with a `PolicyProvenance` or receipt mapping. Its runtime,
I/O, and dataset-manifest IDs/digests are then exposed in `metadata.extra` for
binding validation; leaving it unset preserves the dependency-light default.

Passing `--plan` to `openvla_oft.eval` selects the shared native evaluator
(`evaluation.native_vla_eval`) instead of the upstream loop, so the sealed
schedule, receipt identities, arrow-free observation contract, and common
environment-step budget are actually consumed at rollout time.

Run the dependency-free contract check with:

```powershell
python -m vla_benchmarking.libero.finetuned_vlas.openvla_oft.cli --dry-run
```

The native OpenVLA import occurs only in `OpenVLAOFTAdapter.load`.
