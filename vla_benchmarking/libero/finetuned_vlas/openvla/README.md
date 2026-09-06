# Original OpenVLA LIBERO adapter

This package is intentionally separate from `openvla_oft`. It implements the
official original OpenVLA inference path for
`openvla/openvla-7b-finetuned-libero-spatial`:

- one 224×224 agentview image, no proprioception;
- prompt `In: What action should the robot take to <task>?\nOut:`;
- official 0.9-area center crop and resize;
- `AutoProcessor` + `AutoModelForVision2Seq.predict_action` with
  `unnorm_key="libero_spatial"` and greedy decoding;
- one 7-D action, with official LIBERO gripper normalization/binarization and
  sign inversion.

The model is loaded lazily by `OpenVLAAdapter.load`; tests and imports do not
download checkpoints or require a GPU.
