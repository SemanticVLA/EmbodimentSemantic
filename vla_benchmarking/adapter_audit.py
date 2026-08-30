#!/usr/bin/env python3
"""Fail-closed audit of the action-side SmolVLA LoRA adapter.

The training contract intentionally permits LoRA only on the action expert and
the state/action projection layers.  This module is shared by training
postconditions and graph evaluation so a checkpoint cannot be accepted merely
because it has a plausible-looking PEFT config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ACTION_SIDE_TARGET_REGEX = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
)
REQUIRED_ACTION_MODULES = frozenset({
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
})
_TARGET_RE = re.compile(r"^" + ACTION_SIDE_TARGET_REGEX + r"$")
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


def validate_expected_inventory(value: dict[str, Any]) -> dict[str, Any]:
    """Validate an in-memory or serialized live-policy inventory."""
    if not isinstance(value, dict) or value.get("schema_version") != EXPECTED_INVENTORY_SCHEMA_VERSION:
        raise ValueError("expected LoRA inventory schema is not supported")
    if value.get("inventory_sha256") != inventory_sha256(value):
        raise ValueError("expected LoRA inventory digest is invalid")
    if value.get("target_regex") != ACTION_SIDE_TARGET_REGEX or value.get("peft_type") != "LORA" or int(value.get("rank", -1)) != 16:
        raise ValueError("expected LoRA inventory does not use the sealed action-side rank-16 contract")
    if value.get("modules_to_save") not in ([], None):
        raise ValueError("expected LoRA inventory modules_to_save is not empty")
    if value.get("base_policy_revision") != PINNED_BASE_POLICY_REVISION:
        raise ValueError("expected LoRA inventory base revision is not pinned")
    effective = value.get("effective_peft_config")
    if effective != {
        "peft_type": "LORA", "r": 16,
        "target_modules": ACTION_SIDE_TARGET_REGEX, "modules_to_save": [],
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
    by_module: dict[str, set[str]] = {}
    for name in trainable:
        match = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", name)
        if not match:
            raise ValueError("expected LoRA inventory contains a non-LoRA trainable name")
        module = name[:match.start()]
        if module not in modules or not _TARGET_RE.fullmatch(module):
            raise ValueError("expected LoRA inventory contains a non-action trainable module")
        by_module.setdefault(module, set()).add(match.group(1))
    if any(by_module.get(module) != {"A", "B"} for module in modules):
        raise ValueError("expected LoRA inventory does not contain exactly one A/B pair per module")
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


def _build_expected_inventory_from_model(model: Any, *, base_policy: str | Path = "") -> dict[str, Any]:
    """Build the exact matched/trainable inventory from a wrapped live model.

    This function is deliberately independent of LeRobot constructors so unit
    tests can provide a tiny fake wrapped model.  Production callers must use
    ``build_expected_inventory`` which loads and wraps the pinned LeRobot
    SmolVLA policy first.
    """
    named_modules = list(model.named_modules())
    matched = sorted({
        _strip_model_prefix(name)
        for name, module in named_modules
        if name and _TARGET_RE.fullmatch(_strip_model_prefix(name))
    })
    if not matched:
        raise ValueError("live LoRA-wrapped policy matched no action-side modules")
    if not REQUIRED_ACTION_MODULES.issubset(matched):
        raise ValueError(f"live policy is missing required action/state modules: {sorted(REQUIRED_ACTION_MODULES - set(matched))}")
    if not any(name.startswith("model.vlm_with_expert.lm_expert.") and name.endswith(".q_proj") for name in matched):
        raise ValueError("live policy is missing lm_expert q_proj coverage")
    if not any(name.startswith("model.vlm_with_expert.lm_expert.") and name.endswith(".v_proj") for name in matched):
        raise ValueError("live policy is missing lm_expert v_proj coverage")

    parameters = list(model.named_parameters())
    trainable = []
    trainable_numel = 0
    for raw_name, parameter in parameters:
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        name = _strip_model_prefix(raw_name)
        match = re.search(r"\.lora_([AB])(?:\.[^.]+)?\.weight$", name)
        if not match:
            raise ValueError(f"unexpected trainable non-LoRA parameter in live policy: {raw_name}")
        module = name[:match.start()]
        if module not in matched or not _TARGET_RE.fullmatch(module):
            raise ValueError(f"unexpected trainable module in live policy: {raw_name}")
        trainable.append(name)
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
        "schema_version": EXPECTED_INVENTORY_SCHEMA_VERSION,
        "base_policy": str(Path(base_policy).expanduser().resolve()) if base_policy else None,
        "target_regex": ACTION_SIDE_TARGET_REGEX,
        "peft_type": "LORA",
        "rank": 16,
        "modules_to_save": [],
        "matched_module_names": matched,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": len(trainable),
        "total_parameter_count": len(parameters),
        "trainable_parameter_numel": trainable_numel,
        "total_parameter_numel": sum(_numel(parameter) for _, parameter in parameters),
        "no_vision_backbone_trainables": True,
    }
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


def _load_and_wrap_pinned_smolvla(base_policy: str | Path) -> Any:
    """Load/wrap SmolVLA using the installed, pinned LeRobot runtime.

    LeRobot has moved the SmolVLA class between releases.  We try the known
    local class locations but never fall back to a Hub/Transformers model: a
    compute-node preflight must fail closed if the pinned LeRobot API is absent.
    """
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
        # This is exactly the production CLI path: LeRobot merges the only
        # explicit override (--peft.r=16) with SmolVLA's own default target
        # regex and PEFT method.  Avoid constructing a hand-written config,
        # which can change task_type/alpha semantics across releases.
        wrapped = wrap(peft_cli_overrides={"method_type": "LORA", "r": 16})
        resolved = _effective_peft_config(wrapped)
        if resolved != {
            "peft_type": "LORA", "r": 16,
            "target_modules": ACTION_SIDE_TARGET_REGEX, "modules_to_save": [],
        }:
            raise RuntimeError(f"resolved LeRobot PEFT config differs from the sealed action-side config: {resolved!r}")
        return wrapped
    except Exception as exc:  # pragma: no cover - depends on compute runtime
        raise RuntimeError("pinned LeRobot SmolVLA could not be wrapped with action-side LoRA") from exc


def build_expected_inventory(base_policy: str | Path) -> dict[str, Any]:
    """Load the exact local LeRobot policy and return its sealed LoRA inventory."""
    model = _load_and_wrap_pinned_smolvla(base_policy)
    value = _build_expected_inventory_from_model(model, base_policy=base_policy)
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


def write_expected_inventory(base_policy: str | Path, output: str | Path) -> dict[str, Any]:
    value = build_expected_inventory(base_policy)
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


def _normalise_module_key(key: str) -> tuple[str, str]:
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
    if not module or not _TARGET_RE.fullmatch(module):
        raise ValueError(
            f"adapter tensor targets a non-action-side module: {key}; "
            f"expected {ACTION_SIDE_TARGET_REGEX}"
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
        raise ValueError("modules_to_save must be empty for action-side LoRA")
    if config.get("target_modules") != ACTION_SIDE_TARGET_REGEX:
        raise ValueError("adapter target_modules is not the sealed action-side target regex")
    expected = None
    if expected_inventory is not None:
        expected = load_expected_inventory(expected_inventory) if isinstance(expected_inventory, (str, Path)) else validate_expected_inventory(expected_inventory)
    elif require_expected_inventory:
        raise ValueError("sealed expected live-policy LoRA inventory is required")

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised on compute node
        raise RuntimeError("safetensors is required to audit a LoRA checkpoint") from exc

    inventory: list[dict[str, Any]] = []
    nonzero_b = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = sorted(str(key) for key in handle.keys())
        if not keys:
            raise ValueError("adapter safetensors contains no tensors")
        for key in keys:
            module, side = _normalise_module_key(key)
            tensor = handle.get_tensor(key)
            item = {"key": key, "module": module, "side": side, "shape": _shape(tensor)}
            inventory.append(item)
            if side == "B" and _nonzero(tensor):
                nonzero_b += 1
    by_module: dict[str, dict[str, dict[str, Any]]] = {}
    for item in inventory:
        sides = by_module.setdefault(item["module"], {})
        if item["side"] in sides:
            raise ValueError(f"adapter contains duplicate LoRA {item['side']} tensors for {item['module']}")
        sides[item["side"]] = item
    if not by_module or any(set(sides) != {"A", "B"} for sides in by_module.values()):
        raise ValueError("every action-side module must contain exactly one LoRA A and one LoRA B tensor")
    missing_action = sorted(REQUIRED_ACTION_MODULES - set(by_module))
    if missing_action:
        raise ValueError(f"adapter is missing required action/state LoRA modules: {missing_action}")
    if not any(module.startswith("model.vlm_with_expert.lm_expert.") and module.endswith(".q_proj") for module in by_module):
        raise ValueError("adapter is missing lm_expert q_proj LoRA coverage")
    if not any(module.startswith("model.vlm_with_expert.lm_expert.") and module.endswith(".v_proj") for module in by_module):
        raise ValueError("adapter is missing lm_expert v_proj LoRA coverage")
    for module, sides in by_module.items():
        shape_a, shape_b = sides["A"]["shape"], sides["B"]["shape"]
        if (
            len(shape_a) != 2 or len(shape_b) != 2 or shape_a[0] != 16 or shape_b[1] != 16
        ):
            raise ValueError(f"LoRA rank/shape mismatch for {module}; expected 2-D A/B with A[0]=B[1]=16")
    if require_nonzero_lora_b and nonzero_b < 1:
        raise ValueError("adapter has no nonzero lora_B tensor after training/smoke")
    if expected is not None:
        expected_modules = set(expected.get("matched_module_names", ()))
        actual_modules = set(by_module)
        if actual_modules != expected_modules:
            missing = sorted(expected_modules - actual_modules)
            extra = sorted(actual_modules - expected_modules)
            raise ValueError(f"adapter module set differs from live expected inventory (missing={missing}, extra={extra})")

    inventory_digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "target_regex": ACTION_SIDE_TARGET_REGEX,
        "peft_type": "LORA",
        "rank": 16,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="checkpoint pretrained_model directory")
    parser.add_argument("--generate-expected", action="store_true", help="build the live-policy inventory before training")
    parser.add_argument("--base-policy", default=None, help="local pinned SmolVLA policy used for --generate-expected")
    parser.add_argument("--expected-inventory", default=None, help="sealed live-policy inventory JSON required for checkpoint audit")
    parser.add_argument("--output", default=None, help="audit JSON path (defaults to checkpoint/adapter_audit.json)")
    parser.add_argument("--require-nonzero-lora-b", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.generate_expected:
        if not args.base_policy or not args.output:
            parser.error("--generate-expected requires --base-policy and --output")
        write_expected_inventory(args.base_policy, args.output)
        print(Path(args.output).expanduser().resolve())
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --generate-expected is used")
    evidence = audit_adapter_checkpoint(
        args.checkpoint,
        require_nonzero_lora_b=args.require_nonzero_lora_b,
        expected_inventory=args.expected_inventory,
        require_expected_inventory=True,
    )
    output = Path(args.output).expanduser().resolve() if args.output else Path(args.checkpoint).expanduser().resolve() / "adapter_audit.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
