# Retired arrow controller policies

`v9d_rgbd_region_grasp_search.json` is the only executable arrow controller
policy. The documents in this directory are preserved for provenance and
historical analysis, but the runtime loader rejects them before LIBERO
environment construction.

Retired experiments: v9 patient/base and control, v9a bounded grasp search,
v9b residual micro-correction, v9c combined, v9e micro-correction, v9f height
sweep, v9g post-lift retention, v9h source approach, v9i support-plane
placement, v9j combined evidence repair, and the optional v10 ZeroGrasp grasp
and reconstruction-placement policies. They were rejected as active defaults
because paired canaries did not beat the v9d control, or (for ZeroGrasp) the
external model/runtime and frame calibration were not validated as a better
end-to-end policy. Fine-tuned VLA/LoRA configurations are outside this
directory and remain unchanged.

The active v9d semantic configuration hash is:

`60f4f5f9ecfde7b4830f376ab06cfc706e2ef175d86817c42a0adb7cddd46c0c`
