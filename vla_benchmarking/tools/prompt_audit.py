"""JSONL prompt audit logging for VLA context ablations."""

from __future__ import annotations

import json
import hashlib
import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from vla_benchmarking.evaluation.scene_graph_formats import GRAPH_TOKENIZER_CONTRACT

SEALED_BASE_POLICY_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _graph_profile_enabled() -> bool:
    """Identify graph profiles without changing historical prompt defaults."""
    values = (
        os.environ.get("TRAINING_PROFILE", ""),
        os.environ.get("PROFILE", ""),
        os.environ.get("CONTEXT_FORMAT", ""),
    )
    return any(value.strip().lower() in {"graph_treatment", "arrow_graph_treatment", "target_natural_v1"} for value in values)


def _flatten_input_ids(input_ids: Any) -> list[int]:
    if input_ids is None:
        return []
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        return list(input_ids[0])
    return list(input_ids)


def _tokenizer_asset_inventory(root: Path) -> dict[str, str]:
    """Hash every staged tokenizer file, including config/vocabulary files."""
    asset = root / "tokenizer"
    if not asset.is_dir():
        return {}
    return {
        path.relative_to(asset).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(asset.rglob("*"))
        if path.is_file()
    }


def _tokenizer_provenance(root: Path) -> tuple[dict[str, Any] | None, Path]:
    path = root / "tokenizer_provenance.json"
    if not path.is_file():
        return None, path
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"tokenizer provenance is unreadable: {path}") from exc


def _assert_local_tokenizer_provenance(root: Path) -> dict[str, Any]:
    provenance, provenance_path = _tokenizer_provenance(root)
    if not isinstance(provenance, dict) or provenance.get("materialized") is not True:
        raise RuntimeError(f"graph tokenizer local snapshot is not materialized: {provenance_path}")
    if provenance.get("model_id") != GRAPH_TOKENIZER_CONTRACT["model_id"]:
        raise RuntimeError("graph tokenizer provenance model identity mismatch")
    if provenance.get("revision") != GRAPH_TOKENIZER_CONTRACT["revision"]:
        raise RuntimeError("graph tokenizer provenance revision is not pinned")
    if not isinstance(provenance.get("vocab_sha256"), str) or not provenance["vocab_sha256"]:
        raise RuntimeError("graph tokenizer provenance lacks a sealed vocabulary digest")
    inventory = _tokenizer_asset_inventory(root)
    if inventory != provenance.get("files"):
        raise RuntimeError("graph tokenizer local asset file inventory/hash drifted")
    expected_tree = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if provenance.get("tree_sha256") != expected_tree:
        raise RuntimeError("graph tokenizer local asset tree digest mismatch")
    return provenance


def _snapshot_inventory(root: Path) -> dict[str, str]:
    """Return the complete immutable policy-file inventory (excluding manifest)."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "base_snapshot_manifest.json" and ".cache" not in path.parts
    }


def _validate_snapshot_manifest(root: Path, *, require_derived: bool = False) -> dict[str, Any]:
    """Authenticate a policy snapshot and reject missing, extra, or changed files."""
    manifest_path = root / "base_snapshot_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"base policy snapshot manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"base policy snapshot manifest is unreadable: {manifest_path}") from exc
    revision = manifest.get("revision")
    files = manifest.get("files")
    if not isinstance(revision, str) or not revision.strip() or not isinstance(files, dict) or not files:
        raise RuntimeError("base policy snapshot manifest lacks a pinned revision/file inventory")
    if any(not isinstance(name, str) or name.startswith(("/", "\\")) or ".." in Path(name).parts for name in files):
        raise RuntimeError("base policy snapshot manifest contains an unsafe file name")
    actual = _snapshot_inventory(root)
    if set(actual) != set(files):
        raise RuntimeError("base policy snapshot file inventory drifted")
    for name, expected_hash in files.items():
        if not isinstance(expected_hash, str) or actual.get(name) != expected_hash:
            raise RuntimeError(f"base policy snapshot file hash mismatch: {name}")
    derived_from = manifest.get("derived_from")
    if require_derived and (not isinstance(derived_from, str) or not derived_from):
        raise RuntimeError("derived graph policy snapshot lacks authenticated source binding")
    return {
        "manifest": manifest,
        "revision": revision,
        "files": dict(files),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _validate_derived_source_binding(target: Path, source: Path, source_evidence: dict[str, Any]) -> None:
    """Prove a graph snapshot is derived from exactly this authenticated source."""
    if source_evidence.get("revision") != SEALED_BASE_POLICY_REVISION:
        raise RuntimeError("graph policy source revision is not the sealed SmolVLA revision")
    target_evidence = _validate_snapshot_manifest(target, require_derived=True)
    manifest = target_evidence["manifest"]
    if Path(str(manifest.get("derived_from"))).expanduser().resolve() != source.resolve():
        raise RuntimeError("graph policy snapshot source binding differs from requested base policy")
    if manifest.get("source_manifest_sha256") != source_evidence["sha256"]:
        raise RuntimeError("graph policy snapshot source manifest digest differs")
    if manifest.get("source_revision") != source_evidence["revision"]:
        raise RuntimeError("graph policy snapshot source revision differs")
    if manifest.get("source_files") != source_evidence["files"]:
        raise RuntimeError("graph policy snapshot source file inventory differs")
    expected_source_tree = hashlib.sha256(
        json.dumps(source_evidence["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("source_tree_sha256") != expected_source_tree:
        raise RuntimeError("graph policy snapshot source tree digest differs")

    source_files = source_evidence["files"]
    target_files = target_evidence["files"]
    allowed_changes = {"policy_preprocessor.json", "tokenizer_provenance.json"}
    for name, source_hash in source_files.items():
        if name in allowed_changes or name.startswith("tokenizer/"):
            continue
        if target_files.get(name) != source_hash:
            raise RuntimeError(f"graph policy snapshot changed non-processor source file: {name}")
    allowed_extra = {name for name in target_files if name.startswith("tokenizer/") or name in allowed_changes}
    if set(target_files) - set(source_files) - allowed_extra:
        raise RuntimeError("graph policy snapshot contains unexpected derived files")


class PromptAuditLogger:
    """Append prompt metadata, and optional tokenizer stats, to JSONL."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str] | None,
        *,
        model_id: str | None,
        enabled: bool | None = None,
        token_audit: bool | None = None,
    ):
        self.graph_profile = _graph_profile_enabled()
        self.enabled = (True if self.graph_profile else _bool_env("PROMPT_AUDIT", False)) if enabled is None else enabled
        self.token_audit = (True if self.graph_profile else _bool_env("TOKEN_AUDIT", False)) if token_audit is None else token_audit
        self.strict_token_audit = self.graph_profile or _bool_env("STRICT_TOKEN_AUDIT", False)
        self.model_id = model_id
        self.tokenizer_model_id = self._resolve_tokenizer_model_id(model_id)
        self.max_length_override = self._resolve_max_length_override(model_id)
        self._tokenizer = None
        self._tokenizer_vocab_sha256: str | None = None
        self._tokenizer_tree_sha256: str | None = None
        self._tokenizer_error: str | None = None
        self._warned_tokenizer_error = False
        self._fh = None

        if not self.enabled:
            self.path = None
            return

        if output_dir is None:
            output_dir = "."
        path = Path(output_dir) / "prompt_audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8", buffering=1)

    @staticmethod
    def _resolve_tokenizer_model_id(model_id: str | None) -> str | None:
        override = os.environ.get("TOKENIZER_MODEL") or os.environ.get("TOKENIZER_MODEL_ID")
        if override:
            return override
        if _graph_profile_enabled():
            # Graph evaluation/training must name the local asset explicitly;
            # never silently resolve the Hub model from a checkpoint id.
            return None
        if model_id and "smolvla" in model_id.lower():
            return "HuggingFaceTB/SmolVLM2-500M-Instruct"
        return model_id

    @staticmethod
    def _resolve_max_length_override(model_id: str | None) -> int | None:
        if model_id and "smolvla" in model_id.lower() and _graph_profile_enabled():
            return 96
        value = os.environ.get("TOKENIZER_MAX_LENGTH")
        if value:
            return int(value)
        if model_id and "smolvla" in model_id.lower():
            return 96 if _graph_profile_enabled() else 48
        return None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _load_tokenizer(self):
        if self._tokenizer is not None or self._tokenizer_error is not None:
            return self._tokenizer
        if not self.tokenizer_model_id:
            self._tokenizer_error = "graph mode requires TOKENIZER_MODEL pointing to the local tokenizer snapshot"
            return None

        try:
            from transformers import AutoTokenizer
            if self.graph_profile:
                local_root = Path(self.tokenizer_model_id).expanduser().resolve()
                if not local_root.is_dir():
                    raise RuntimeError("TOKENIZER_MODEL is not a local tokenizer directory")
                # The policy checkpoint owns tokenizer_provenance.json one
                # level above its tokenizer directory.
                self._tokenizer_tree_sha256 = None
                provenance, _ = _tokenizer_provenance(local_root.parent)
                if not isinstance(provenance, dict) or provenance.get("materialized") is not True:
                    raise RuntimeError("local tokenizer provenance is absent or not materialized")
                self._tokenizer_tree_sha256 = provenance.get("tree_sha256")
                self._tokenizer = AutoTokenizer.from_pretrained(local_root, trust_remote_code=True, local_files_only=True)
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_id, trust_remote_code=True)
            self._tokenizer_vocab_sha256 = _vocab_sha256(self._tokenizer)
        except Exception as exc:
            self._tokenizer_error = f"{type(exc).__name__}: {exc}"
            return None

        return self._tokenizer

    def _token_stats(self, prompt: str) -> dict[str, Any]:
        if not self.token_audit:
            return {
                "token_audit_enabled": False,
                "token_count_before_truncation": None,
                "token_count_after_truncation": None,
                "truncation_occurred": None,
                "decoded_truncated_prompt": None,
                "tokenizer_error": None,
            }

        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            return {
                "token_audit_enabled": True,
                "token_count_before_truncation": None,
                "token_count_after_truncation": None,
                "truncation_occurred": None,
                "decoded_truncated_prompt": None,
                "tokenizer_error": self._tokenizer_error,
            }

        effective_prompt = prompt
        if self.graph_profile and not effective_prompt.endswith("\n"):
            effective_prompt += "\n"
        encoded = tokenizer(effective_prompt, add_special_tokens=True, truncation=False)
        before_ids = _flatten_input_ids(encoded.get("input_ids"))
        before_mask = _flatten_input_ids(encoded.get("attention_mask"))
        if not before_mask:
            before_mask = [1] * len(before_ids)
        max_len = self.max_length_override or getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 1_000_000:
            max_len = None

        if max_len is None:
            after_ids = before_ids
        else:
            truncated = tokenizer(
                effective_prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len,
            )
            after_ids = _flatten_input_ids(truncated.get("input_ids"))

        try:
            decoded = tokenizer.decode(after_ids, skip_special_tokens=False)
        except Exception:
            decoded = None

        return {
            "token_audit_enabled": True,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_model_max_length": max_len,
            "tokenizer_truncation_side": getattr(tokenizer, "truncation_side", None),
            "tokenizer_vocab_sha256": self._tokenizer_vocab_sha256,
            "tokenizer_tree_sha256": self._tokenizer_tree_sha256,
            "effective_prompt_sha256": hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
            "input_ids": before_ids,
            "attention_mask": before_mask,
            "input_ids_attention_mask_sha256": hashlib.sha256(json.dumps({"input_ids": before_ids, "attention_mask": before_mask}, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "token_count_before_truncation": len(before_ids),
            "token_count_after_truncation": len(after_ids),
            "truncation_occurred": len(after_ids) < len(before_ids),
            "decoded_truncated_prompt": decoded,
            "tokenizer_error": None,
        }

    def log(
        self,
        *,
        prompt: str,
        task_text: str,
        context_mode: str,
        context_format: str,
        task_id: int | None,
        env_step: int | None,
        relations_generated: int,
        relations_retained: int,
        relations: list[tuple[str, str, str]] | None = None,
        reset_sequence: int | None = None,
    ) -> None:
        if not self.enabled or self._fh is None:
            return

        token_stats = self._token_stats(prompt)
        decoded = token_stats.get("decoded_truncated_prompt")
        if self.strict_token_audit:
            if token_stats.get("tokenizer_error"):
                raise RuntimeError(f"strict tokenizer audit failed: {token_stats['tokenizer_error']}")
            if token_stats.get("truncation_occurred"):
                raise RuntimeError("strict tokenizer audit failed: graph prompt was truncated")
            if decoded is not None and task_text not in decoded:
                raise RuntimeError("strict tokenizer audit failed: task instruction was not retained")
        record = {
            "format_name": context_format,
            "context_mode": context_mode,
            "task_id": task_id,
            "episode_id": None,
            "env_step": env_step,
            "raw_prompt": prompt,
            "task_text": task_text,
            "character_count": len(prompt),
            "number_of_relations_generated": relations_generated,
            "number_of_relations_retained": relations_retained,
            "task_instruction_present_after_truncation": (
                None if decoded is None else task_text in decoded
            ),
        }
        if self.graph_profile:
            triplets = [list(item) for item in (relations or [])]
            record.update({
                "triplets": triplets,
                "triplet_sha256": hashlib.sha256(
                    json.dumps(triplets, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "reset_sequence": reset_sequence,
            })
        record.update(token_stats)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _vocab_sha256(tokenizer: Any) -> str:
    """Hash the complete tokenizer vocabulary, including token ids."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise RuntimeError("tokenizer does not expose get_vocab(); identity cannot be audited")
    vocab = get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise RuntimeError("tokenizer vocabulary is empty or invalid")
    payload = json.dumps(sorted((str(token), int(index)) for token, index in vocab.items()), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_graph_prompts(
    prompts: list[str] | set[str],
    *,
    tokenizer: Any,
    task_text_by_prompt: dict[str, str] | None = None,
    contract: dict[str, Any] = GRAPH_TOKENIZER_CONTRACT,
    allow_local_model_path: bool = False,
) -> dict[str, Any]:
    """Fail closed if graph prompts violate the sealed tokenizer contract."""
    if contract != GRAPH_TOKENIZER_CONTRACT:
        raise ValueError("graph tokenizer contract differs from the canonical contract")
    unique_prompts = sorted(set(prompts))
    if not unique_prompts:
        raise ValueError("graph tokenizer audit received no prompts")
    vocab_sha256 = _vocab_sha256(tokenizer)
    tokenizer_model = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", None)
    if tokenizer_model and tokenizer_model != contract["model_id"]:
        if not allow_local_model_path or not Path(str(tokenizer_model)).exists():
            raise RuntimeError(
                f"tokenizer/model identity mismatch: expected {contract['model_id']!r}, got {tokenizer_model!r}"
            )
    encoded_lengths: list[int] = []
    token_records: list[dict[str, Any]] = []
    for prompt in unique_prompts:
        # SmolVLA's serialized newline step appends exactly one newline before
        # TokenizerProcessorStep; audit the effective string, not the dataset
        # spelling stored in the task column.
        effective_prompt = prompt if prompt.endswith("\n") else prompt + "\n"
        encoded = tokenizer(effective_prompt, add_special_tokens=True, truncation=False)
        before_ids = _flatten_input_ids(encoded.get("input_ids") if hasattr(encoded, "get") else None)
        before_mask = _flatten_input_ids(encoded.get("attention_mask") if hasattr(encoded, "get") else None)
        if not before_mask:
            before_mask = [1] * len(before_ids)
        if len(before_ids) > int(contract["max_length"]):
            raise RuntimeError(f"graph prompt exceeds {contract['max_length']} tokens: {prompt!r}")
        truncated = tokenizer(
            effective_prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=int(contract["max_length"]),
        )
        after_ids = _flatten_input_ids(truncated.get("input_ids") if hasattr(truncated, "get") else None)
        if after_ids != before_ids:
            raise RuntimeError(f"graph tokenizer truncated a prompt: {prompt!r}")
        try:
            decoded = tokenizer.decode(before_ids, skip_special_tokens=False)
        except Exception as exc:
            raise RuntimeError("graph tokenizer could not decode audited prompt") from exc
        task_text = (task_text_by_prompt or {}).get(prompt)
        if contract["task_instruction_must_be_retained"] and task_text and task_text not in decoded:
            raise RuntimeError(f"graph tokenizer dropped the task instruction: {task_text!r}")
        encoded_lengths.append(len(before_ids))
        token_records.append({"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "input_ids": before_ids, "attention_mask": before_mask})
    token_digest = hashlib.sha256(
        json.dumps(token_records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "contract": dict(contract),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_model_id": tokenizer_model,
        "vocab_sha256": vocab_sha256,
        "tokenizer_vocab_sha256": vocab_sha256,
        "prompt_count": len(unique_prompts),
        "max_observed_tokens": max(encoded_lengths),
        "add_special_tokens": True,
        "truncation_checked": True,
        "prompt_token_ids_sha256": token_digest,
        "prompt_token_audit_sha256": token_digest,
    }


def audit_historical_prompt_parity(
    prompts: list[str] | set[str], *, tokenizer: Any, require_all_short: bool = False
) -> dict[str, Any]:
    """Prove that short historical prompts keep identical ids/masks at 48/96.

    This is a diagnostic comparability check only: graph prompts still use the
    sealed 96-token contract.  Padding is deliberately ``longest`` to mirror
    LeRobot's SmolVLA processor semantics.
    """
    records: list[dict[str, Any]] = []
    for prompt in sorted(set(prompts)):
        effective = prompt if prompt.endswith("\n") else prompt + "\n"
        encoded: dict[int, tuple[list[int], list[int]]] = {}
        for ceiling in (48, 96):
            try:
                value = tokenizer(effective, add_special_tokens=True, truncation=False, max_length=ceiling, padding="longest")
            except TypeError:
                value = tokenizer(effective, add_special_tokens=True, truncation=False, max_length=ceiling)
            ids = _flatten_input_ids(value.get("input_ids") if hasattr(value, "get") else None)
            mask = _flatten_input_ids(value.get("attention_mask") if hasattr(value, "get") else None)
            if not mask:
                mask = [1] * len(ids)
            encoded[ceiling] = (ids, mask)
        if len(encoded[48][0]) > 48 and require_all_short:
            raise RuntimeError(f"historical task prompt exceeds the 48-token control budget: {prompt!r}")
        if len(encoded[48][0]) <= 48 and encoded[48] != encoded[96]:
            raise RuntimeError(f"historical short-prompt ids/masks differ between 48 and 96: {prompt!r}")
        if len(encoded[48][0]) <= 48:
            records.append({"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "input_ids_sha256": hashlib.sha256(json.dumps(encoded[48], separators=(",", ":")).encode("utf-8")).hexdigest()})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"checked": len(records), "prompt_ids_masks_sha256": digest, "ceilings": [48, 96], "padding": "longest"}


def audit_graph_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    base_policy: str | os.PathLike[str],
    dataset_roots: list[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Audit manifest/source prompts and, when available, both dataset trees."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("tokenizer_contract") != GRAPH_TOKENIZER_CONTRACT:
        raise ValueError("graph manifest tokenizer contract is not canonical")
    expected_digest = hashlib.sha256(
        json.dumps(GRAPH_TOKENIZER_CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if manifest.get("tokenizer_contract_sha256") != expected_digest:
        raise ValueError("graph manifest tokenizer contract digest is invalid")
    prompts: set[str] = set()
    historical_prompts: set[str] = set()
    task_text_by_prompt: dict[str, str] = {}
    for task in manifest.get("tasks", []):
        task_text = task.get("task_text")
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("graph manifest task text is missing for historical tokenizer parity")
        historical_prompts.add(task_text)
        for demo in task.get("demos", []):
            for record in demo.get("graph_prompt_records", []):
                prompt = record.get("prompt")
                if prompt:
                    expected_prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    if record.get("prompt_sha256") != expected_prompt_sha:
                        raise ValueError("graph manifest prompt digest does not match prompt bytes")
                    triplets = record.get("triplets", [])
                    expected_triplet_sha = hashlib.sha256(
                        json.dumps(triplets, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    ).hexdigest()
                    if record.get("triplet_sha256") != expected_triplet_sha:
                        raise ValueError("graph manifest triplet digest does not match triplet bytes")
                    prompts.add(prompt)
                    task_text_by_prompt[prompt] = str(task.get("task_text", ""))
    # Older manifests store only prompt hashes. In that case the dataset roots
    # are required to supply the exact task strings.
    for root in dataset_roots or []:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(repo_id=f"local/{Path(root).name}", root=Path(root))
            for index in range(len(dataset)):
                prompt = dataset[index].get("task")
                if isinstance(prompt, str):
                    prompts.add(prompt)
        except Exception as exc:
            raise RuntimeError(f"could not scan graph dataset root {root}: {exc}") from exc
    tokenizer, processor_evidence = _load_effective_graph_tokenizer(base_policy, require_runtime=True)
    audit = audit_graph_prompts(
        prompts,
        tokenizer=tokenizer,
        task_text_by_prompt=task_text_by_prompt,
        allow_local_model_path=True,
    )
    if len(historical_prompts) != len(manifest.get("tasks", [])) or len(historical_prompts) != 10:
        raise ValueError("graph manifest must provide exactly 10 unique task prompts for historical parity")
    audit["historical_prompt_parity"] = audit_historical_prompt_parity(
        historical_prompts, tokenizer=tokenizer, require_all_short=True
    )
    audit["base_policy"] = str(Path(base_policy).expanduser().resolve())
    audit["manifest"] = str(manifest_file.resolve())
    audit["processor_evidence"] = processor_evidence
    return audit


def _read_tokenizer_processor_config(
    base_policy: str | os.PathLike[str], *, require_local: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Read and validate the serialized step consumed by LeRobot.

    ``policy_preprocessor.json`` is deliberately checked before importing any
    Transformers processor.  SmolVLA policy snapshots contain this LeRobot
    pipeline but do not necessarily contain a Transformers ``preprocessor_config``.
    """
    root = Path(base_policy).expanduser().resolve()
    preprocessor_path = root / "policy_preprocessor.json"
    if not preprocessor_path.is_file():
        raise RuntimeError(f"LeRobot policy_preprocessor.json is missing: {preprocessor_path}")
    try:
        payload = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LeRobot policy_preprocessor.json is unreadable: {exc}") from exc
    steps = payload.get("steps")
    tokenizer_steps = [step for step in steps or [] if isinstance(step, dict) and step.get("registry_name") == "tokenizer_processor"]
    if len(tokenizer_steps) != 1:
        raise RuntimeError("LeRobot policy preprocessor must contain exactly one tokenizer_processor step")
    config = tokenizer_steps[0].get("config")
    if not isinstance(config, dict):
        raise RuntimeError("LeRobot tokenizer_processor config is missing")
    expected = {
        "max_length": GRAPH_TOKENIZER_CONTRACT["max_length"],
        "truncation": True,
        "padding": GRAPH_TOKENIZER_CONTRACT["padding"],
        "padding_side": GRAPH_TOKENIZER_CONTRACT["padding_side"],
        "task_key": "task",
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise RuntimeError(
                f"effective LeRobot tokenizer config mismatch for {key}: expected {wanted!r}, got {config.get(key)!r}"
            )
    tokenizer_name = str(config.get("tokenizer_name", ""))
    if not tokenizer_name:
        raise RuntimeError("LeRobot tokenizer_processor tokenizer_name is missing")
    if require_local:
        tokenizer_path = Path(tokenizer_name).expanduser()
        if not tokenizer_path.is_absolute() or tokenizer_path.resolve() != (root / "tokenizer").resolve() or not tokenizer_path.is_dir():
            raise RuntimeError("graph tokenizer must resolve to the exact local policy/tokenizer path")
    newline_steps = [step for step in steps or [] if isinstance(step, dict) and step.get("registry_name") == "smolvla_new_line_processor"]
    if len(newline_steps) != 1:
        raise RuntimeError("SmolVLA policy preprocessor must contain exactly one smolvla_new_line_processor step")
    return preprocessor_path, config


def _processor_pipeline(base_policy: Path) -> tuple[Any | None, str | None]:
    """Load the actual LeRobot processor pipeline when the runtime is present."""
    try:
        from lerobot.processor import PolicyProcessorPipeline
    except Exception as exc:
        return None, f"LeRobot PolicyProcessorPipeline unavailable: {type(exc).__name__}: {exc}"
    try:
        try:
            pipeline = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=base_policy,
                config_filename="policy_preprocessor.json",
                local_files_only=True,
            )
        except TypeError as exc:
            # Older LeRobot releases do not expose local_files_only on this
            # method; they still accept the explicit config filename.
            if "local_files_only" not in str(exc):
                raise
            pipeline = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=base_policy,
                config_filename="policy_preprocessor.json",
            )
        return pipeline, None
    except Exception as exc:
        raise RuntimeError(f"could not load LeRobot policy processor from {base_policy}: {exc}") from exc


def _pipeline_steps(pipeline: Any) -> list[Any]:
    for name in ("steps", "processors", "pipeline"):
        value = getattr(pipeline, name, None)
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


def _step_config(step: Any) -> dict[str, Any]:
    for name in ("config", "kwargs"):
        value = getattr(step, name, None)
        if isinstance(value, dict):
            return value
    to_dict = getattr(step, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
            if isinstance(value, dict):
                return value.get("config", value)
        except Exception:
            pass
    get_config = getattr(step, "get_config", None)
    if callable(get_config):
        try:
            value = get_config()
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


def _runtime_tokenizer(pipeline: Any) -> tuple[Any | None, dict[str, Any]]:
    """Extract the tokenizer step/tokenizer from the live LeRobot pipeline."""
    for step in _pipeline_steps(pipeline):
        registry = getattr(step, "registry_name", None)
        if registry != "tokenizer_processor" and "tokenizer" not in type(step).__name__.lower():
            continue
        config = _step_config(step)
        tokenizer = getattr(step, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(step, "_tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(step, "input_tokenizer", None)
        return tokenizer, {
            "step_class": type(step).__name__,
            "step_registry_name": registry,
            "runtime_config": config,
            "tokenizer_model_id": getattr(tokenizer, "name_or_path", None) if tokenizer is not None else None,
            "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        }
    return None, {"step_class": None, "step_registry_name": None, "runtime_config": {}}


def validate_serialized_graph_preprocessor(
    policy_dir: str | os.PathLike[str],
    *,
    require_runtime: bool = False,
    require_snapshot_manifest: bool | None = None,
) -> dict[str, Any]:
    """Validate the effective graph preprocessor, failing closed on 48 tokens.

    The serialized step is mandatory.  When LeRobot is installed (as it is on
    the compute node), ``require_runtime=True`` additionally loads the exact
    ``PolicyProcessorPipeline`` used by training and compares its tokenizer
    step.  Historical 48-token snapshots therefore cannot pass graph
    preflight merely because a policy CLI override says 96.
    """
    if require_snapshot_manifest is None:
        require_snapshot_manifest = require_runtime
    root = Path(policy_dir).expanduser().resolve()
    preprocessor_path, config = _read_tokenizer_processor_config(
        root, require_local=bool(require_snapshot_manifest or require_runtime)
    )
    local_provenance = None
    if config.get("tokenizer_name") != GRAPH_TOKENIZER_CONTRACT["model_id"]:
        local_provenance = _assert_local_tokenizer_provenance(root)
    snapshot_manifest_path = root / "base_snapshot_manifest.json"
    snapshot_evidence: dict[str, Any] = {"path": str(snapshot_manifest_path), "present": snapshot_manifest_path.is_file()}
    if require_snapshot_manifest:
        evidence = _validate_snapshot_manifest(root, require_derived=True)
        snapshot_evidence.update({"revision": evidence["revision"], "sha256": evidence["sha256"]})
        source_name = evidence["manifest"].get("derived_from")
        if not isinstance(source_name, str):
            raise RuntimeError("graph policy snapshot source binding is missing")
        _validate_derived_source_binding(root, Path(source_name), _validate_snapshot_manifest(Path(source_name)))
    pipeline, runtime_error = (
        _processor_pipeline(root) if require_runtime else (None, "runtime processor check not requested")
    )
    runtime_tokenizer = None
    runtime_evidence: dict[str, Any] = {"available": pipeline is not None}
    if pipeline is None:
        runtime_evidence["error"] = runtime_error
        if require_runtime:
            raise RuntimeError(runtime_error or "LeRobot policy processor is unavailable")
    else:
        runtime_tokenizer, step_evidence = _runtime_tokenizer(pipeline)
        runtime_evidence.update(step_evidence)
        if not runtime_evidence.get("step_class"):
            raise RuntimeError("LeRobot policy processor has no tokenizer_processor step")
        newline_runtime = [
            step for step in _pipeline_steps(pipeline)
            if getattr(step, "registry_name", None) == "smolvla_new_line_processor"
            or "newline" in type(step).__name__.lower()
            or "new_line" in type(step).__name__.lower()
        ]
        if len(newline_runtime) != 1:
            raise RuntimeError("live SmolVLA policy processor must contain exactly one newline task step")
        runtime_evidence["newline_step_class"] = type(newline_runtime[0]).__name__
        runtime_config = runtime_evidence.get("runtime_config") or {}
        runtime_name = runtime_config.get("tokenizer_name")
        if runtime_name != str((root / "tokenizer").resolve()):
            raise RuntimeError(
                "live graph tokenizer must resolve to the exact local policy/tokenizer path"
            )
        for key, wanted in (
            ("max_length", 96),
            ("truncation", True),
            ("padding", GRAPH_TOKENIZER_CONTRACT["padding"]),
            ("padding_side", GRAPH_TOKENIZER_CONTRACT["padding_side"]),
            ("task_key", "task"),
        ):
            if runtime_config.get(key) != wanted:
                raise RuntimeError(
                    f"live LeRobot tokenizer step mismatch for {key}: expected {wanted!r}, got {runtime_config.get(key)!r}"
                )
        runtime_model = runtime_evidence.get("tokenizer_model_id")
        if runtime_model:
            runtime_path = Path(str(runtime_model)).expanduser()
            if not runtime_path.is_absolute() or runtime_path.resolve() != (root / "tokenizer").resolve():
                raise RuntimeError("live graph tokenizer resolved outside the local sealed snapshot")
    return {
        "preprocessor_path": str(preprocessor_path),
        "preprocessor_sha256": hashlib.sha256(preprocessor_path.read_bytes()).hexdigest(),
        "tokenizer_name": config["tokenizer_name"],
        "max_length": config["max_length"],
        "truncation": config["truncation"],
        "task_key": config["task_key"],
        "tokenizer_revision": GRAPH_TOKENIZER_CONTRACT["revision"],
        "tokenizer_provenance_sha256": (
            hashlib.sha256((root / "tokenizer_provenance.json").read_bytes()).hexdigest()
            if local_provenance is not None else None
        ),
        "runtime_processor": runtime_evidence,
        "snapshot_manifest": snapshot_evidence,
    }


def retarget_graph_checkpoint_preprocessor(policy_dir: str | os.PathLike[str]) -> Path:
    """Bind a checkpoint's tokenizer step to its own absolute tokenizer asset."""
    root = Path(policy_dir).expanduser().resolve()
    preprocessor = root / "policy_preprocessor.json"
    tokenizer = root / "tokenizer"
    if not tokenizer.is_dir():
        raise RuntimeError(f"graph checkpoint tokenizer asset is missing: {tokenizer}")
    try:
        payload = json.loads(preprocessor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"graph checkpoint preprocessor is unreadable: {preprocessor}") from exc
    steps = [step for step in payload.get("steps", []) if isinstance(step, dict) and step.get("registry_name") == "tokenizer_processor"]
    if len(steps) != 1:
        raise RuntimeError("graph checkpoint must contain exactly one tokenizer_processor step")
    steps[0].setdefault("config", {})["tokenizer_name"] = str(tokenizer.resolve())
    preprocessor.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return preprocessor


def _load_effective_graph_tokenizer(
    base_policy: str | os.PathLike[str],
    *,
    require_runtime: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load the tokenizer named by LeRobot's serialized policy preprocessor.

    SmolVLA snapshots do not necessarily contain Transformers processor files.
    The serialized ``policy_preprocessor.json`` is the source of truth used by
    ``PolicyProcessorPipeline.from_pretrained``; we validate it first and then
    instantiate the exact tokenizer named by its ``tokenizer_processor`` step.
    """
    root = Path(base_policy).expanduser().resolve()
    processor_evidence = validate_serialized_graph_preprocessor(root, require_runtime=require_runtime)
    config = {"tokenizer_name": processor_evidence["tokenizer_name"]}
    runtime = processor_evidence["runtime_processor"]
    tokenizer = None
    if runtime.get("available"):
        # Prefer the tokenizer object held by the exact live step.  This keeps
        # token ids tied to LeRobot's processor rather than a guessed VLM
        # ``AutoProcessor`` reconstruction.
        for step in _pipeline_steps(_processor_pipeline(root)[0]):
            if "tokenizer" in type(step).__name__.lower() or getattr(step, "registry_name", None) == "tokenizer_processor":
                tokenizer = (
                    getattr(step, "tokenizer", None)
                    or getattr(step, "_tokenizer", None)
                    or getattr(step, "input_tokenizer", None)
                )
                if tokenizer is not None:
                    break
    try:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer_name = config["tokenizer_name"]
            tokenizer_path = Path(tokenizer_name)
            if not tokenizer_path.is_absolute():
                tokenizer_path = root / tokenizer_path
            if not tokenizer_path.is_dir():
                raise RuntimeError("graph tokenizer is not a local staged asset")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True, local_files_only=True
            )
    except Exception as exc:
        raise RuntimeError(f"could not load tokenizer named by LeRobot processor: {exc}") from exc
    processor_evidence["source"] = "lerobot_policy_processor_pipeline"
    processor_evidence["tokenizer_class"] = type(tokenizer).__name__
    processor_evidence["tokenizer_model_id"] = getattr(tokenizer, "name_or_path", None)
    processor_evidence["vocab_sha256"] = _vocab_sha256(tokenizer)
    processor_evidence["tokenizer_vocab_sha256"] = processor_evidence["vocab_sha256"]
    provenance, _ = _tokenizer_provenance(root)
    if isinstance(provenance, dict) and processor_evidence["vocab_sha256"] != provenance.get("vocab_sha256"):
        raise RuntimeError("effective graph tokenizer vocabulary digest differs from the sealed local asset")
    return tokenizer, processor_evidence


def prepare_graph_policy_snapshot(
    base_policy: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create an immutable hard-linked base snapshot with a graph-only processor override."""
    source = Path(base_policy).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    source_preprocessor = source / "policy_preprocessor.json"
    if not source.is_dir() or not source_preprocessor.is_file():
        raise RuntimeError(f"base policy/preprocessor is missing: {source}")
    source_evidence = _validate_snapshot_manifest(source)
    # Graph derivation is sealed to the same immutable SmolVLA base used by
    # the historical experiments; a merely self-consistent arbitrary snapshot
    # must not become a new training lineage.
    if source_evidence["revision"] != SEALED_BASE_POLICY_REVISION:
        raise RuntimeError("base policy snapshot revision is not the sealed SmolVLA revision")
    if target.exists():
        existing = target / "policy_preprocessor.json"
        if not existing.is_file():
            raise RuntimeError(f"graph policy snapshot already exists without preprocessor: {target}")
        _validate_derived_source_binding(target, source, source_evidence)
        data = json.loads(existing.read_text(encoding="utf-8"))
        steps = [step for step in data.get("steps", []) if step.get("registry_name") == "tokenizer_processor"]
        if (
            len(steps) != 1
            or steps[0].get("config", {}).get("max_length") != 96
            or steps[0].get("config", {}).get("tokenizer_name") != str((target / "tokenizer").resolve())
        ):
            raise RuntimeError(f"existing graph policy snapshot is not sealed to max_length=96: {target}")
        return {"path": str(target), "source": str(source), "reused": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".pending")
    if staging.exists():
        raise RuntimeError(f"stale graph policy staging directory exists: {staging}")
    try:
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if ".cache" in relative.parts:
                continue
            destination = staging / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                if relative.as_posix() in {"base_snapshot_manifest.json", "tokenizer_provenance.json"} or (
                    relative.parts and relative.parts[0] == "tokenizer"
                ):
                    # The derived manifest is written independently below;
                    # never hard-link the source manifest, or rewriting the
                    # derived record would mutate the authenticated source.
                    # Provenance is also regenerated below and must be a copy.
                    # Tokenizer assets must be copied too: save_pretrained()
                    # materializes files in-place and a hardlink would mutate
                    # the authenticated source snapshot.
                    if relative.as_posix() == "base_snapshot_manifest.json":
                        continue
                    shutil.copy2(path, destination)
                    continue
                if relative.as_posix() == "policy_preprocessor.json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    steps = [step for step in data.get("steps", []) if step.get("registry_name") == "tokenizer_processor"]
                    if len(steps) != 1:
                        raise RuntimeError("base policy has no unique tokenizer_processor step")
                    source_config = steps[0].get("config", {})
                    for key in ("truncation", "padding", "padding_side", "task_key"):
                        if source_config.get(key) != {
                            "truncation": True,
                            "padding": GRAPH_TOKENIZER_CONTRACT["padding"],
                            "padding_side": GRAPH_TOKENIZER_CONTRACT["padding_side"],
                            "task_key": "task",
                        }[key]:
                            raise RuntimeError(f"base LeRobot tokenizer {key} semantics differ from the sealed historical processor")
                    steps[0]["config"]["max_length"] = 96
                    steps[0]["config"]["truncation"] = source_config["truncation"]
                    steps[0]["config"]["padding"] = source_config["padding"]
                    steps[0]["config"]["padding_side"] = source_config["padding_side"]
                    # LeRobot resolves relative tokenizer assets from the
                    # policy snapshot.  This prevents an accidental Hub
                    # lookup at train/eval time.
                    steps[0]["config"]["tokenizer_name"] = str((target / "tokenizer").resolve())
                    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                else:
                    try:
                        os.link(path, destination)
                    except OSError:
                        shutil.copy2(path, destination)
        # Materialize the exact pinned tokenizer locally when Transformers is
        # available on the compute host.  Local preparation without the ML
        # runtime still emits explicit non-materialized provenance; strict
        # graph preflight rejects that state rather than falling back to Hub.
        tokenizer_dir = staging / "tokenizer"
        provenance: dict[str, Any] = {
            "materialized": False,
            "model_id": GRAPH_TOKENIZER_CONTRACT["model_id"],
            "revision": GRAPH_TOKENIZER_CONTRACT["revision"],
            "files": {},
            "tree_sha256": hashlib.sha256(b"{}").hexdigest(),
        }
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                GRAPH_TOKENIZER_CONTRACT["model_id"],
                revision=GRAPH_TOKENIZER_CONTRACT["revision"],
                trust_remote_code=True,
            )
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_dir)
            inventory = _tokenizer_asset_inventory(staging)
            provenance.update({
                "materialized": bool(inventory),
                "files": inventory,
                "tokenizer_class": type(tokenizer).__name__,
                "vocab_sha256": _vocab_sha256(tokenizer),
                "tree_sha256": hashlib.sha256(
                    json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            })
        except Exception as exc:
            # Offline staging can carry an already materialized, pinned asset
            # from a trusted source snapshot.  Recompute its inventory here;
            # never infer provenance from an arbitrary directory alone.
            existing_provenance, _ = _tokenizer_provenance(staging)
            inventory = _tokenizer_asset_inventory(staging)
            if (
                isinstance(existing_provenance, dict)
                and existing_provenance.get("materialized") is True
                and existing_provenance.get("model_id") == GRAPH_TOKENIZER_CONTRACT["model_id"]
                and existing_provenance.get("revision") == GRAPH_TOKENIZER_CONTRACT["revision"]
                and existing_provenance.get("files") == inventory
            ):
                provenance = dict(existing_provenance)
                provenance["tree_sha256"] = hashlib.sha256(
                    json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            else:
                provenance["error"] = f"{type(exc).__name__}: {exc}"
        # The source is authenticated both before and after derivation.  This
        # catches accidental writes through shared inodes as well as external
        # mutation during a long tokenizer materialization step.
        source_after = _validate_snapshot_manifest(source)
        if source_after != source_evidence:
            raise RuntimeError("authenticated base snapshot changed during graph derivation")
        (staging / "tokenizer_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Bind the derived snapshot with a fresh file inventory while retaining
        # the original model revision and explicit source provenance.
        files = {}
        for path in sorted(staging.rglob("*")):
            if (
                path.is_file()
                and path.name != "base_snapshot_manifest.json"
                and ".cache" not in path.parts
            ):
                files[path.relative_to(staging).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        revision = None
        source_manifest = source / "base_snapshot_manifest.json"
        if source_manifest.is_file():
            try:
                revision = json.loads(source_manifest.read_text(encoding="utf-8")).get("revision")
            except (OSError, json.JSONDecodeError):
                revision = None
        manifest = {
            "schema_version": 2,
            "revision": revision,
            "derived_from": str(source),
            "derived_from_preprocessor_sha256": hashlib.sha256(source_preprocessor.read_bytes()).hexdigest(),
            "source_manifest_sha256": source_evidence["sha256"],
            "source_revision": source_evidence["revision"],
            "source_files": source_evidence["files"],
            "source_tree_sha256": hashlib.sha256(
                json.dumps(source_evidence["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "graph_tokenizer_max_length": 96,
            "files": files,
        }
        (staging / "base_snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"path": str(target), "source": str(source), "reused": False}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Strict tokenizer audit for sealed graph prompts")
    parser.add_argument("--graph-manifest")
    parser.add_argument("--base-policy")
    parser.add_argument("--dataset-root", action="append", default=[])
    parser.add_argument("--prepare-graph-policy", nargs=2, metavar=("BASE", "OUTPUT"))
    parser.add_argument("--verify-graph-policy")
    parser.add_argument("--verify-graph-checkpoint")
    parser.add_argument("--retarget-graph-checkpoint")
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    if args.prepare_graph_policy:
        print(json.dumps(prepare_graph_policy_snapshot(*args.prepare_graph_policy), indent=2, sort_keys=True))
        return 0
    if args.verify_graph_policy:
        report = validate_serialized_graph_preprocessor(args.verify_graph_policy, require_runtime=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.verify_graph_checkpoint:
        report = validate_serialized_graph_preprocessor(
            args.verify_graph_checkpoint,
            require_runtime=True,
            require_snapshot_manifest=False,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.retarget_graph_checkpoint:
        print(retarget_graph_checkpoint_preprocessor(args.retarget_graph_checkpoint))
        return 0
    if not args.graph_manifest or not args.base_policy:
        parser.error("--graph-manifest and --base-policy are required unless --prepare-graph-policy is used")
    report = audit_graph_manifest(args.graph_manifest, base_policy=args.base_policy, dataset_roots=args.dataset_root)
    if args.audit_output:
        output = Path(args.audit_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
