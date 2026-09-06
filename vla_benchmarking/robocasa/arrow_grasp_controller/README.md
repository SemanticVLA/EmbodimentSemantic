# RoboCasa arrow grasp controller

This folder is the self-contained RoboCasa copy of the known-good LIBERO arrow
grasp controller. It preserves the controller pipeline and policy constants,
and adapts only the RoboCasa camera/frame names, role-bbox arrow input, source
noun, official evaluator, and PandaOmron composite action boundary.

The `legacy_engine` directory is not the removed RoboCasa shortcut. It is the
same low-level module used by the canonical LIBERO controller and is mirrored
locally so RoboCasa has no runtime import from the LIBERO package. The earlier
RoboCasa path that called only that low-level routine has been replaced by the
full high-level controller in `controller/runner.py`.
