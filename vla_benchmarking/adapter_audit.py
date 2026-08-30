#!/usr/bin/env python3
"""Fail-closed audit of a versioned SmolVLA LoRA adapter.

The historical action-only contract remains schema-1 compatible. The new
action_visual_lora_v1 contract adds only the VLM connector and late vision
q/v projections; it never permits full-weight trainables or VLM text weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from lora_finetuning_policy import (
    ACTION_ONLY_LORA_V1,
    ACTION_SIDE_TARGET_REGEX,
    ACTION_VISUAL_LORA_V1,
    ACTION_VISUAL_TARGET_REGEX,
    DEFAULT_FINETUNING_POLICY,
    FinetuningPolicy,
    LORA_RANK,
    get_policy,
    policy_from_inventory,
)

# Public aliases retained for historical schema-1 callers and fixtures.
REQUIRED_ACTION_MODULES = frozenset(get_policy(ACTION_ONLY_LORA_V1).required_module_names)
_TARGET_RE = get_policy(ACTION_ONLY_LORA_V1).target_pattern
EXPECTED_INVENTORY_SCHEMA_VERSION = 1
PINNED_BASE_POLICY_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def _strip_model_prefix(value: str) -> str:
    """Normalize the names emitted by PEFT and by a wrapped LeRobot policy."""
    value = str(value)
    while value.startswith("base_model.model.") or value.startswith("base_model."):
        value = value.split(".", 2)[-1] if value.startswith("base_model.") else value
        # The first branch above intentionally handles both prefixes without
        # assuming whether LeRobot added one or two wrapper modules.
        if value.startswith("model."):
            break
    return value


def _canonical_inventory(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("inventory_sha256", None)
    return body


def inventory_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_canonical_inventory(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _resolve_inventory_policy(value: dict[str, Any]) -> FinetuningPolicy:
    try:
        return policy_from_inventory(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected LoRA inventory policy is invalid: {exc}") from exc


def _validate_module_coverage(modules: list[str], policy: FinetuningPolicy) -> None:
    module_set = set(modules)
    missing_names = sorted(set(policy.required_module_names) - module_set)
    if missing_names:
        raise ValueError(f"expected {policy.policy_id} inventory is missing required modules: {missing_names}")
    for pattern in policy.required_module_patterns:
        if not any(re.fullmatch(pattern, module) for module in modules):
            raise ValueError(f"expected {policy.policy_id} inventory is missing required module coverage: {pattern}")


def validate_expected_inventory(value: dict[str, Any]) -> dict[str, Any]:
    """Validate an in-memory or serialized live-policy inventory.

    Missing ``finetuning_policy_id`` is intentionally interpreted as the
    historical action-only policy so old manifests and checkpoints remain
    readable without rewriting their immutable evidence.
    """
    if not isinstance(value, dict):
        raise ValueError("expected LoRA inventory schema is not supported")
    policy = _resolve_inventory_policy(value)
    if value.get("schema_version") != policy.inventory_schema_version:
        raise ValueError("expected LoRA inventory schema is not supported")
    if value.get("inventory_sha256") != inventory_sha256(value):
        raise ValueError("expected LoRA inventory digest is invalid")
    if value.get("target_regex") != policy.target_regex or value.get("peft_type") != "LORA" or int(value.get("rank", -1)) != LORA_RANK:
        raise ValueError(f"expected LoRA inventory does not use the sealed {policy.policy_id} rank-{LORA_RANK} contract")
    if value.get("modules_to_save") not in ([], None):
        raise ValueError("expected LoRA inventory modules_to_save is not empty")
    if value.get("base_policy_revision") != PINNED_BASE_POLICY_REVISION:
        raise ValueError("expected LoRA inventory base revision is not pinned")
    effective = value.get("effective_peft_config")
    if effective != {
        "peft_type": "LORA", "r": LORA_RANK,
        "target_modules": policy.target_regex, "modules_to_save": [],
    }:
        raise ValueError("expected LoRA inventory lacks the exact resolved LeRobot PEFT config")
    modules = value.get("matched_module_names")
    trainable = value.get("trainable_parameter_names")
    if not isinstance(modules, list) or not modules or not all(isinstance(item, str) for item in modules):
        raise ValueError("expected LoRA inventory has no complete matched module set")
    if modules != sorted(set(modules)):
        raise ValueError("expected LoRA inventory module set is not canonical")
    if not isinstance(trainable, list) or trainable != sorted(set(trainable)):
        raise ValueError("expected LoRA inventory trainable names are not canonical")
    if int(value.get("trainable_parameter_count", -1)) != len(trainable):
        raise ValueError("expected LoRA inventory trainable count is inconsistent")
    _validate_module_coverage(modules, policy)
    by_module: dict[str, set[str]] = {}
    for name in trainable:
        match = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", name)
        if not match:
            raise ValueError("expected LoRA inventory contains a non-LoRA trainable name")
        module = name[:match.start()]
        if module not in modules or not policy.target_pattern.fullmatch(module):
            raise ValueError(f"expected LoRA inventory contains a module outside {policy.policy_id}")
        by_module.setdefault(module, set()).add(match.group(1))
    if any(by_module.get(module) != {"A", "B"} for module in modules):
        raise ValueError("expected LoRA inventory does not contain exactly one A/B pair per module")
    if value.get("finetuning_policy_id", DEFAULT_FINETUNING_POLICY) != policy.policy_id:
        raise ValueError("expected LoRA inventory policy identity is inconsistent")
    metadata = value.get("finetuning_policy")
    if metadata is not None and metadata != policy.to_metadata():
        raise ValueError("expected LoRA inventory policy metadata does not match the canonical policy")
    if policy.policy_id == ACTION_VISUAL_LORA_V1:
        if value.get("no_full_weight_trainables") is not True or value.get("base_parameters_frozen") is not True:
            raise ValueError("visual LoRA inventory must assert frozen base parameters and no full-weight trainables")
        if len(modules) != policy.expected_target_count:
            raise ValueError(f"{policy.policy_id} expected {policy.expected_target_count} target modules, got {len(modules)}")
        if int(value.get("trainable_parameter_numel", -1)) != policy.expected_trainable_numel:
            raise ValueError(f"{policy.policy_id} expected {policy.expected_trainable_numel} trainable parameters, got {value.get('trainable_parameter_numel')}")
        shapes = value.get("trainable_parameter_shapes")
        if not isinstance(shapes, dict) or set(shapes) != set(trainable):
            raise ValueError("visual LoRA inventory must seal shapes for every canonical trainable parameter")
        shape_numel = 0
        for name in trainable:
            if name != _canonical_lora_name(name):
                raise ValueError("visual LoRA inventory trainable keys are not canonical")
            shape = shapes[name]
            if not isinstance(shape, list) or not shape or not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise ValueError(f"visual LoRA inventory has an invalid shape for {name}")
            item_numel = 1
            for dim in shape:
                item_numel *= dim
            shape_numel += item_numel
        if int(value.get("trainable_tensor_count", -1)) != len(trainable) or shape_numel != int(value["trainable_parameter_numel"]):
            raise ValueError("visual LoRA inventory tensor count/numel seal is inconsistent")
    return value


def load_expected_inventory(path: str | Path) -> dict[str, Any]:
    """Load the sealed live-policy inventory and reject any mutation."""
    target = Path(path).expanduser().resolve()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"expected LoRA inventory is unreadable: {target}") from exc
    return validate_expected_inventory(value)


def _numel(parameter: Any) -> int:
    value = getattr(parameter, "numel", None)
    if callable(value):
        return int(value())
    shape = _shape(parameter)
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _canonical_lora_name(name: str) -> str:
    """Normalize wrapper/default-adapter variants to one persisted key."""
    normalized = _strip_model_prefix(name)
    match = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", normalized)
    if not match:
        raise ValueError(f"parameter is not a LoRA A/B weight: {name}")
    return f"{normalized[:match.start()]}.lora_{match.group(1)}.default.weight"


def _build_expected_inventory_from_model(
    model: Any,
    *,
    base_policy: str | Path = "",
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Build the exact matched/trainable inventory from a wrapped live model.

    This function is deliberately independent of LeRobot constructors so unit
    tests can provide a tiny fake wrapped model.  Production callers must use
    ``build_expected_inventory`` which loads and wraps the pinned LeRobot
    SmolVLA policy first.
    """
    policy = get_policy(policy_id)
    target_re = policy.target_pattern
    named_modules = list(model.named_modules())
    matched = sorted({
        _strip_model_prefix(name)
        for name, module in named_modules
        if name and target_re.fullmatch(_strip_model_prefix(name))
    })
    if not matched:
        raise ValueError("live LoRA-wrapped policy matched no action-side modules")
    _validate_module_coverage(matched, policy)

    parameters = list(model.named_parameters())
    trainable = []
    trainable_shapes: dict[str, list[int]] = {}
    trainable_numel = 0
    for raw_name, parameter in parameters:
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        name = _strip_model_prefix(raw_name)
        match = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", name)
        if not match:
            raise ValueError(f"unexpected trainable non-LoRA parameter in live policy: {raw_name}")
        module = name[:match.start()]
        if module not in matched or not target_re.fullmatch(module):
            raise ValueError(f"unexpected trainable module in {policy.policy_id}: {raw_name}")
        canonical_name = _canonical_lora_name(name)
        if canonical_name in trainable_shapes:
            raise ValueError(f"live LoRA policy contains duplicate trainable parameter: {raw_name}")
        trainable.append(canonical_name)
        trainable_shapes[canonical_name] = _shape(parameter)
        trainable_numel += _numel(parameter)
    trainable = sorted(set(trainable))
    if not trainable:
        raise ValueError("live LoRA-wrapped policy has no trainable parameters")
    trainable_by_module: dict[str, set[str]] = {}
    for name in trainable:
        match = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", name)
        assert match is not None  # checked above
        trainable_by_module.setdefault(name[:match.start()], set()).add(match.group(1))
    incomplete = sorted(module for module in matched if trainable_by_module.get(module) != {"A", "B"})
    if incomplete:
        raise ValueError(f"live LoRA policy has incomplete A/B coverage: {incomplete}")
    expected = {
        "schema_version": policy.inventory_schema_version,
        "finetuning_policy_id": policy.policy_id,
        "finetuning_policy": policy.to_metadata(),
        "base_policy": str(Path(base_policy).expanduser().resolve()) if base_policy else None,
        "base_policy_revision": PINNED_BASE_POLICY_REVISION,
        "target_regex": policy.target_regex,
        "peft_type": "LORA",
        "rank": LORA_RANK,
        "modules_to_save": [],
        "matched_module_names": matched,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": len(trainable),
        "total_parameter_count": len(parameters),
        "trainable_parameter_numel": trainable_numel,
        "total_parameter_numel": sum(_numel(parameter) for _, parameter in parameters),
    }
    if policy.policy_id == ACTION_VISUAL_LORA_V1:
        expected["trainable_parameter_shapes"] = {name: trainable_shapes[name] for name in trainable}
        expected["trainable_tensor_count"] = len(trainable)
        expected["no_full_weight_trainables"] = True
        expected["base_parameters_frozen"] = True
    else:
        # Historical schema-1 field retained for old action-only inventories.
        expected["no_vision_backbone_trainables"] = True
    if policy.policy_id == ACTION_VISUAL_LORA_V1:
        if len(matched) != policy.expected_target_count:
            raise ValueError(f"live {policy.policy_id} matched {len(matched)} modules; expected {policy.expected_target_count}")
        if trainable_numel != policy.expected_trainable_numel:
            raise ValueError(f"live {policy.policy_id} has {trainable_numel} trainable parameters; expected {policy.expected_trainable_numel}")
    expected["inventory_sha256"] = inventory_sha256(expected)
    return expected


def _effective_peft_config(model: Any) -> dict[str, Any]:
    """Serialize the resolved PEFT config produced by LeRobot's wrapper."""
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, dict) or not configs:
        raise RuntimeError("LeRobot PEFT wrapper did not expose a resolved peft_config")
    config = next(iter(configs.values()))
    peft_type = getattr(config, "peft_type", None)
    peft_type = getattr(peft_type, "value", peft_type)
    target_modules = getattr(config, "target_modules", None)
    if isinstance(target_modules, (set, tuple, list)):
        target_modules = sorted(str(value) for value in target_modules)
    modules_to_save = getattr(config, "modules_to_save", None)
    if isinstance(modules_to_save, (set, tuple)):
        modules_to_save = sorted(str(value) for value in modules_to_save)
    elif modules_to_save is None:
        modules_to_save = []
    return {
        "peft_type": str(peft_type).upper(),
        "r": int(getattr(config, "r", -1)),
        "target_modules": target_modules,
        "modules_to_save": list(modules_to_save),
    }


def _load_and_wrap_pinned_smolvla(
    base_policy: str | Path,
    *,
    policy_id: str | None = None,
) -> Any:
    """Load/wrap SmolVLA using the installed, pinned LeRobot runtime.

    LeRobot has moved the SmolVLA class between releases.  We try the known
    local class locations but never fall back to a Hub/Transformers model: a
    compute-node preflight must fail closed if the pinned LeRobot API is absent.
    """
    policy = get_policy(policy_id)
    base = str(Path(base_policy).expanduser().resolve())
    if not Path(base).is_dir():
        raise ValueError(f"local base policy does not exist: {base}")
    try:
        import peft  # noqa: F401  # imported to fail clearly before model construction
    except ImportError as exc:  # pragma: no cover - compute-node boundary
        raise RuntimeError("peft is required to build the expected LoRA inventory") from exc
    candidates = (
        ("lerobot.policies.smolvla", "SmolVLAPolicy"),
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
        ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLA"),
        ("lerobot.policies.smolvla.model", "SmolVLAPolicy"),
    )
    errors: list[str] = []
    model = None
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            loader = getattr(cls, "from_pretrained", None)
            if not callable(loader):
                raise TypeError(f"{class_name} has no from_pretrained")
            try:
                model = loader(pretrained_name_or_path=base, local_files_only=True)
            except TypeError:
                # Older pinned LeRobot releases accept the path positionally;
                # both forms remain local-only and therefore cannot float to
                # an unpinned Hub revision.
                try:
                    model = loader(base, local_files_only=True)
                except TypeError:
                    model = loader(base)
            break
        except Exception as exc:  # pragma: no cover - depends on compute runtime
            errors.append(f"{module_name}.{class_name}: {exc}")
    if model is None:
        raise RuntimeError("could not load pinned LeRobot SmolVLA locally; " + " | ".join(errors))
    try:
        wrap = getattr(model, "wrap_with_peft", None)
        if not callable(wrap):
            raise RuntimeError("pinned LeRobot policy has no production wrap_with_peft entrypoint")
        # Mirror the production CLI path while making the policy's target
        # regex explicit. This prevents the visual policy from silently
        # resolving to SmolVLA's historical action-only default.
        wrapped = wrap(peft_cli_overrides={
            "method_type": "LORA", "r": LORA_RANK,
            "target_modules": policy.target_regex,
        })
        resolved = _effective_peft_config(wrapped)
        expected = {
            "peft_type": "LORA", "r": LORA_RANK,
            "target_modules": policy.target_regex, "modules_to_save": [],
        }
        if resolved != expected:
            raise RuntimeError(f"resolved LeRobot PEFT config differs from the sealed {policy.policy_id} config: {resolved!r}")
        return wrapped
    except Exception as exc:  # pragma: no cover - depends on compute runtime
        raise RuntimeError("pinned LeRobot SmolVLA could not be wrapped with action-side LoRA") from exc


def build_expected_inventory(
    base_policy: str | Path,
    *,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Load the exact local LeRobot policy and return its sealed LoRA inventory."""
    policy = get_policy(policy_id)
    model = _load_and_wrap_pinned_smolvla(base_policy, policy_id=policy.policy_id)
    value = _build_expected_inventory_from_model(model, base_policy=base_policy, policy_id=policy.policy_id)
    value["effective_peft_config"] = _effective_peft_config(model)
    manifest = Path(base_policy).expanduser().resolve() / "base_snapshot_manifest.json"
    if manifest.is_file():
        try:
            value["base_policy_revision"] = json.loads(manifest.read_text(encoding="utf-8")).get("revision")
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"base policy snapshot manifest is unreadable: {manifest}") from exc
    if value.get("base_policy_revision") != PINNED_BASE_POLICY_REVISION:
        raise ValueError("expected LoRA inventory cannot identify the pinned base revision")
    value["inventory_sha256"] = inventory_sha256(value)
    return value


def write_expected_inventory(
    base_policy: str | Path,
    output: str | Path,
    *,
    policy_id: str | None = None,
) -> dict[str, Any]:
    value = build_expected_inventory(base_policy, policy_id=policy_id)
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_inventory(adapter_dir: str | Path) -> dict[str, str]:
    """Hash every consumed checkpoint file except this audit sidecar."""
    root = Path(adapter_dir).expanduser().resolve()
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "adapter_audit.json"
    }


def checkpoint_tree_sha256(adapter_dir: str | Path) -> str:
    return hashlib.sha256(
        json.dumps(checkpoint_inventory(adapter_dir), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalise_module_key(key: str, policy: FinetuningPolicy | None = None) -> tuple[str, str]:
    """Return (base module, A/B side), rejecting unknown adapter key shapes."""
    value = key
    for prefix in ("base_model.model.", "base_model."):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    match = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", value)
    if not match:
        raise ValueError(f"adapter tensor is not a LoRA A/B weight: {key}")
    module = value[: match.start()]
    policy = policy or get_policy(DEFAULT_FINETUNING_POLICY)
    if not module or not policy.target_pattern.fullmatch(module):
        raise ValueError(
            f"adapter tensor targets a non-action-side module outside {policy.policy_id}: {key}; "
            f"expected {policy.target_regex}"
        )
    return module, match.group(1)


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("adapter tensor has no shape")
    return [int(item) for item in shape]


def _nonzero(value: Any) -> bool:
    # Torch tensors, numpy arrays, and the tiny fake tensors used by tests all
    # support a conversion through ``detach/cpu/numpy`` or ``any``.
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    any_method = getattr(candidate, "any", None)
    if callable(any_method):
        result = any_method()
        return bool(result.item()) if hasattr(result, "item") else bool(result)
    try:
        return any(float(item) != 0.0 for item in candidate.reshape(-1))
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        raise ValueError("cannot inspect lora_B tensor for nonzero values") from exc


def audit_adapter_checkpoint(
    adapter_dir: str | Path,
    *,
    require_nonzero_lora_b: bool = True,
    expected_inventory: str | Path | dict[str, Any] | None = None,
    require_expected_inventory: bool = False,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Audit one real PEFT checkpoint and return sealed evidence.

    ``adapter_dir`` is the checkpoint's ``pretrained_model`` directory.  The
    function deliberately imports safetensors lazily: source-only contract
    tests can import this module without installing the GPU training stack.
    """
    directory = Path(adapter_dir).expanduser().resolve()
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapter_model.safetensors"
    if not config_path.is_file() or not config_path.stat().st_size:
        raise ValueError(f"adapter_config.json is missing or empty: {config_path}")
    if not weights_path.is_file() or not weights_path.stat().st_size:
        raise ValueError(f"adapter_model.safetensors is missing or empty: {weights_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"adapter_config.json is not valid JSON: {config_path}") from exc
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter PEFT type is not LORA")
    if int(config.get("r", -1)) != 16:
        raise ValueError("adapter rank must be exactly 16")
    modules_to_save = config.get("modules_to_save")
    if modules_to_save not in (None, [], ()):
        raise ValueError("modules_to_save must be empty for the selected LoRA policy")
    expected = None
    if expected_inventory is not None:
        expected = load_expected_inventory(expected_inventory) if isinstance(expected_inventory, (str, Path)) else validate_expected_inventory(expected_inventory)
    elif require_expected_inventory:
        raise ValueError("sealed expected live-policy LoRA inventory is required")
    # The policy id is an experiment-side contract, not a PEFT requirement.
    # Prefer the explicit CLI value, then the sealed live inventory, and only
    # finally accept an optional sidecar field if a runtime emitted one.
    config_policy_id = policy_id
    if config_policy_id in (None, "") and expected is not None:
        config_policy_id = expected.get("finetuning_policy_id")
    if config_policy_id in (None, "") and isinstance(config.get("finetuning_policy_id"), str):
        config_policy_id = config["finetuning_policy_id"]
    policy = get_policy(config_policy_id)
    if config.get("target_modules") != policy.target_regex:
        raise ValueError(f"adapter target_modules is not the sealed {policy.policy_id} target regex")

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised on compute node
        raise RuntimeError("safetensors is required to audit a LoRA checkpoint") from exc

    inventory: list[dict[str, Any]] = []
    nonzero_b = 0
    actual_trainable_shapes: dict[str, list[int]] = {}
    actual_trainable_numel = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = sorted(str(key) for key in handle.keys())
        if not keys:
            raise ValueError("adapter safetensors contains no tensors")
        for key in keys:
            module, side = _normalise_module_key(key, policy)
            tensor = handle.get_tensor(key)
            item = {"key": key, "module": module, "side": side, "shape": _shape(tensor)}
            inventory.append(item)
            canonical_key = _canonical_lora_name(key)
            if canonical_key in actual_trainable_shapes:
                raise ValueError(f"adapter contains duplicate canonical trainable tensor: {canonical_key}")
            actual_trainable_shapes[canonical_key] = item["shape"]
            item_numel = 1
            for dim in item["shape"]:
                item_numel *= dim
            actual_trainable_numel += item_numel
            if side == "B" and _nonzero(tensor):
                nonzero_b += 1
    by_module: dict[str, dict[str, dict[str, Any]]] = {}
    for item in inventory:
        sides = by_module.setdefault(item["module"], {})
        if item["side"] in sides:
            raise ValueError(f"adapter contains duplicate LoRA {item['side']} tensors for {item['module']}")
        sides[item["side"]] = item
    if not by_module or any(set(sides) != {"A", "B"} for sides in by_module.values()):
        raise ValueError("every permitted module must contain exactly one LoRA A and one LoRA B tensor")
    _validate_module_coverage(sorted(by_module), policy)
    for module, sides in by_module.items():
        shape_a, shape_b = sides["A"]["shape"], sides["B"]["shape"]
        if (
            len(shape_a) != 2 or len(shape_b) != 2 or shape_a[0] != 16 or shape_b[1] != 16
        ):
            raise ValueError(f"LoRA rank/shape mismatch for {module}; expected 2-D A/B with A[0]=B[1]=16")
    if require_nonzero_lora_b and nonzero_b < 1:
        raise ValueError("adapter has no nonzero lora_B tensor after training/smoke")
    if expected is not None:
        expected_policy = _resolve_inventory_policy(expected)
        if expected_policy.policy_id != policy.policy_id:
            raise ValueError("adapter policy differs from expected live-policy inventory")
        expected_modules = set(expected.get("matched_module_names", ()))
        actual_modules = set(by_module)
        if actual_modules != expected_modules:
            missing = sorted(expected_modules - actual_modules)
            extra = sorted(actual_modules - expected_modules)
            raise ValueError(f"adapter module set differs from live expected inventory (missing={missing}, extra={extra})")
        if policy.policy_id == ACTION_VISUAL_LORA_V1:
            expected_shapes = expected.get("trainable_parameter_shapes")
            if not isinstance(expected_shapes, dict):
                raise ValueError("visual LoRA expected inventory lacks trainable tensor shapes")
            if set(actual_trainable_shapes) != set(expected_shapes):
                missing = sorted(set(expected_shapes) - set(actual_trainable_shapes))
                extra = sorted(set(actual_trainable_shapes) - set(expected_shapes))
                raise ValueError(f"visual LoRA trainable tensor keys differ from expected inventory (missing={missing}, extra={extra})")
            shape_mismatches = {
                name: {"expected": expected_shapes[name], "actual": actual_trainable_shapes[name]}
                for name in expected_shapes
                if actual_trainable_shapes[name] != expected_shapes[name]
            }
            if shape_mismatches:
                raise ValueError(f"visual LoRA trainable tensor shapes differ from expected inventory: {shape_mismatches}")
            expected_count = int(expected.get("trainable_tensor_count", expected.get("trainable_parameter_count", -1)))
            if len(actual_trainable_shapes) != expected_count or actual_trainable_numel != int(expected.get("trainable_parameter_numel", -1)):
                raise ValueError("visual LoRA trainable tensor count/numel differs from expected inventory")

    inventory_digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if policy.policy_id == ACTION_VISUAL_LORA_V1 and len(by_module) != policy.expected_target_count:
        raise ValueError(f"{policy.policy_id} expected {policy.expected_target_count} target modules, got {len(by_module)}")
    report = {
        "schema_version": policy.inventory_schema_version,
        "finetuning_policy_id": policy.policy_id,
        "finetuning_policy": policy.to_metadata(),
        "target_regex": policy.target_regex,
        "peft_type": "LORA",
        "rank": LORA_RANK,
        "modules_to_save": [],
        "tensor_count": len(inventory),
        "tensor_keys": [item["key"] for item in inventory],
        "tensor_shapes": {item["key"]: item["shape"] for item in inventory},
        "tensor_inventory_sha256": inventory_digest,
        "adapter_sha256": _sha256_file(weights_path),
        "checkpoint_inventory": checkpoint_inventory(directory),
        "checkpoint_tree_sha256": checkpoint_tree_sha256(directory),
        "nonzero_lora_b_count": nonzero_b,
        "nonzero_lora_b_required": bool(require_nonzero_lora_b),
        "expected_inventory_sha256": expected.get("inventory_sha256") if expected is not None else None,
        "expected_matched_module_names": sorted(expected.get("matched_module_names", ())) if expected is not None else None,
    }
    if policy.policy_id == ACTION_VISUAL_LORA_V1:
        report["trainable_parameter_shapes"] = actual_trainable_shapes
        report["trainable_tensor_count"] = len(actual_trainable_shapes)
        report["trainable_parameter_numel"] = actual_trainable_numel
        report.update({"no_full_weight_trainables": True, "base_parameters_frozen": True})
    else:
        report["no_vision_backbone_trainables"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="checkpoint pretrained_model directory")
    parser.add_argument("--generate-expected", action="store_true", help="build the live-policy inventory before training")
    parser.add_argument("--base-policy", default=None, help="local pinned SmolVLA policy used for --generate-expected")
    parser.add_argument("--expected-inventory", default=None, help="sealed live-policy inventory JSON required for checkpoint audit")
    parser.add_argument("--finetuning-policy", default=None, help="versioned finetuning policy id (defaults from config or action_only_lora_v1)")
    parser.add_argument("--output", default=None, help="audit JSON path (defaults to checkpoint/adapter_audit.json)")
    parser.add_argument("--require-nonzero-lora-b", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.generate_expected:
        if not args.base_policy or not args.output:
            parser.error("--generate-expected requires --base-policy and --output")
        write_expected_inventory(args.base_policy, args.output, policy_id=args.finetuning_policy)
        print(Path(args.output).expanduser().resolve())
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --generate-expected is used")
    evidence = audit_adapter_checkpoint(
        args.checkpoint,
        require_nonzero_lora_b=args.require_nonzero_lora_b,
        expected_inventory=args.expected_inventory,
        require_expected_inventory=True,
        policy_id=args.finetuning_policy,
    )
    output = Path(args.output).expanduser().resolve() if args.output else Path(args.checkpoint).expanduser().resolve() / "adapter_audit.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
