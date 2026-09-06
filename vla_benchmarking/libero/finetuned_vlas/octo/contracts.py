"""Dependency-light Octo/RLDS and action-space contracts.

The source HDF5 renders are stored in the opposite camera orientation from
LIBERO's policy convention.  Offline serialization therefore applies exactly
one 180-degree image rotation; live adapters must consume the evaluator's
canonical image and must not rotate it a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

OCTO_IMAGE_KEY = "image_primary"
OCTO_LANGUAGE_KEY = "language_instruction"
OCTO_ACTION_DIM = 7
OCTO_ACTION_HORIZON = 4
OCTO_OBSERVATION_WINDOW = 1
FRAME_ORIENTATION_STORED_RAW = "stored_raw"
FRAME_ORIENTATION_LIBERO_CANONICAL = "libero_canonical"
FRAME_ROTATION_OWNER = "octo.dataset.serialize_episode"


def _array(value: Any, *, name: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for Octo array contracts") from exc
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def rotate_stored_frame_180(frame: Any) -> Any:
    """Return a copy rotated 180 degrees in the two spatial axes.

    ``frame`` must be HxWxC.  Importing this module is safe without NumPy;
    NumPy is required only when this operation is actually requested.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required to rotate stored frames") from exc
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"stored frame must be HxWxC with 1, 3, or 4 channels; got {value.shape}")
    return np.flip(value, axis=(0, 1)).copy()


def _validate_action(value: Any, *, name: str) -> Any:
    action = _array(value, name=name).astype("float32", copy=False)
    if action.shape != (OCTO_ACTION_DIM,):
        raise ValueError(f"{name} must have shape (7,), got {action.shape}")
    return action


def _validate_stats(mean: Any, std: Any) -> tuple[Any, Any]:
    mean_arr = _array(mean, name="action_mean").astype("float32", copy=False)
    std_arr = _array(std, name="action_std").astype("float32", copy=False)
    if mean_arr.shape != (6,) or std_arr.shape != (6,):
        raise ValueError("Gaussian action statistics must have shape (6,)")
    if (std_arr <= 0).any():
        raise ValueError("Gaussian action standard deviations must be positive")
    return mean_arr, std_arr


def convert_libero_to_octo_action(action_libero: Any, mean: Any, std: Any) -> Any:
    """Normalize LIBERO action into Octo's six-Gaussian-plus-open format.

    LIBERO's final command uses ``libero_gripper = 1 - 2 * octo_open``.
    Consequently ``octo_open`` is ``(1 - libero_gripper) / 2`` and is not
    normalized with the Gaussian statistics.
    """

    action = _validate_action(action_libero, name="libero_action")
    mean_arr, std_arr = _validate_stats(mean, std)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for action conversion") from exc
    result = np.empty((OCTO_ACTION_DIM,), dtype="float32")
    result[:6] = (action[:6] - mean_arr) / std_arr
    result[6] = (1.0 - action[6]) / 2.0
    return result


def convert_octo_to_libero_action(action_octo: Any, mean: Any, std: Any) -> Any:
    """Unnormalize Octo action and convert its open-gripper command to LIBERO."""

    action = _validate_action(action_octo, name="octo_action")
    mean_arr, std_arr = _validate_stats(mean, std)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for action conversion") from exc
    if not 0.0 <= float(action[6]) <= 1.0:
        raise ValueError("Octo open-gripper command must be in [0, 1]")
    result = np.empty((OCTO_ACTION_DIM,), dtype="float32")
    result[:6] = action[:6] * std_arr + mean_arr
    result[6] = 1.0 - 2.0 * action[6]
    return result


def convert_unnormalized_octo_to_libero_action(action_octo: Any) -> Any:
    """Convert an already-unnormalized Octo action to LIBERO coordinates."""

    action = _validate_action(action_octo, name="unnormalized_octo_action")
    if not 0.0 <= float(action[6]) <= 1.0:
        raise ValueError("Octo open-gripper command must be in [0, 1]")
    result = action.copy()
    result[6] = 1.0 - 2.0 * action[6]
    return result


@dataclass(frozen=True)
class OctoRLDSExample:
    """One serializable RLDS step with source lineage preserved."""

    image_primary: Any
    language_instruction: str
    action: Any
    episode_id: str
    frame_id: int
    is_first: bool = False
    is_last: bool = False
    is_terminal: bool = False
    source_frame_orientation: str = FRAME_ORIENTATION_STORED_RAW

    def to_step(self, *, rotate_stored_frame: bool = True) -> dict[str, Any]:
        if self.source_frame_orientation == FRAME_ORIENTATION_STORED_RAW:
            if not rotate_stored_frame:
                raise ValueError("stored_raw frames must be rotated by the single declared serializer owner")
            frame = rotate_stored_frame_180(self.image_primary)
            rotation_owner = FRAME_ROTATION_OWNER
        elif self.source_frame_orientation == FRAME_ORIENTATION_LIBERO_CANONICAL:
            if rotate_stored_frame:
                raise ValueError("canonical frames must not be rotated a second time")
            frame = self.image_primary
            rotation_owner = "source_manifest"
        else:
            raise ValueError(f"unsupported source frame orientation: {self.source_frame_orientation}")
        action = _validate_action(self.action, name="octo_action")
        if not isinstance(self.language_instruction, str) or not self.language_instruction.strip():
            raise ValueError("language_instruction must be a non-empty string")
        if not self.episode_id or int(self.frame_id) < 0:
            raise ValueError("episode_id and non-negative frame_id are required")
        return {
            "observation": {OCTO_IMAGE_KEY: frame},
            "task": {OCTO_LANGUAGE_KEY: self.language_instruction},
            "action": action,
            "episode_id": str(self.episode_id),
            "frame_id": int(self.frame_id),
            "is_first": bool(self.is_first),
            "is_last": bool(self.is_last),
            "is_terminal": bool(self.is_terminal),
            "frame_provenance": {
                "source_orientation": self.source_frame_orientation,
                "canonical_orientation": FRAME_ORIENTATION_LIBERO_CANONICAL,
                "rotation_owner": rotation_owner,
            },
        }


def serialize_rlds_step(
    *,
    image_primary: Any,
    language_instruction: str,
    action: Any,
    episode_id: str,
    frame_id: int,
    is_first: bool = False,
    is_last: bool = False,
    is_terminal: bool = False,
    rotate_stored_frame: bool = True,
    source_frame_orientation: str = FRAME_ORIENTATION_STORED_RAW,
) -> dict[str, Any]:
    """Serialize one source transition without importing TensorFlow/RLDS."""

    return OctoRLDSExample(
        image_primary=image_primary,
        language_instruction=language_instruction,
        action=action,
        episode_id=episode_id,
        frame_id=frame_id,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        source_frame_orientation=source_frame_orientation,
    ).to_step(rotate_stored_frame=rotate_stored_frame)


def validate_observation(observation: Mapping[str, Any]) -> Any:
    """Validate the native adapter observation and return the primary image."""

    if not isinstance(observation, Mapping):
        raise TypeError("Octo observation must be a mapping")
    if set(observation) - {OCTO_IMAGE_KEY, "timestep", "timestep_pad_mask", "frame_provenance"}:
        raise ValueError("Octo observation may contain only image_primary, timestep, timestep_pad_mask, and frame_provenance")
    provenance = observation.get("frame_provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping) or provenance.get("canonical_orientation") != FRAME_ORIENTATION_LIBERO_CANONICAL:
            raise ValueError("Octo live observation must be in the canonical frame orientation")
        if not str(provenance.get("rotation_owner", "")).strip():
            raise ValueError("Octo canonical observation lacks a rotation owner")
    if OCTO_IMAGE_KEY not in observation:
        raise ValueError("Octo observation is missing image_primary")
    frame = observation[OCTO_IMAGE_KEY]
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("numpy is required for observation validation") from exc
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"image_primary must be HxWxC; got {image.shape}")
    if image.shape[0] != 256 or image.shape[1] != 256:
        raise ValueError(f"Octo LIBERO image must be 256x256; got {image.shape[:2]}")
    return image
