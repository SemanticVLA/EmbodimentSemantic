# RoboCasa VLA integrations

This directory mirrors the LIBERO benchmark layout for future VLA adapters.
The initial RoboCasa PickPlace-21 port intentionally evaluates only the
canonical `arrow_grasp_controller`; no VLA checkpoint or fine-tuning pipeline
is part of the baseline transfer experiment.

When an adapter is added, it must record its checkpoint, processor version,
observation keys, action layout, and target-split provenance alongside results.
