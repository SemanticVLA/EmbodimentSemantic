from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

from scene_graph_formats import (
    GRAPH_TOKENIZER_CONTRACT,
    TARGET_NATURAL_FORMAT,
    format_scene_context,
    format_target_natural_v1,
    normalize_context_format,
)
from prompt_audit import (
    audit_graph_prompts,
    prepare_graph_policy_snapshot,
    validate_serialized_graph_preprocessor,
)


class _FakeTokenizer:
    name_or_path = "HuggingFaceTB/SmolVLM2-500M-Instruct"

    def get_vocab(self):
        return {"<pad>": 0, **{chr(65 + i): i + 1 for i in range(26)}}

    def __call__(self, prompt, *, add_special_tokens, truncation, max_length=None):
        del add_special_tokens
        ids = [ord(char) % 26 for char in prompt]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr((item % 26) + 65) for item in ids)


def test_target_natural_v1_is_target_centric_and_disambiguates_bowls() -> None:
    prompt = format_target_natural_v1(
        "Pick up the target bowl and place it on the plate.",
        [
            ("akita_black_bowl_2", "is_left_of", "plate_1"),
            ("akita_black_bowl_1", "is_left_of", "akita_black_bowl_2"),
            ("akita_black_bowl_1", "is_left_of", "akita_black_bowl_2"),
            ("akita_black_bowl_1", "is_in_front_of", "cookies_1"),
        ],
    )
    assert prompt == (
        "Task: Pick up the target bowl and place it on the plate.\n"
        "Spatial relations: The target black bowl is in front of the cookie box "
        "and is left of the other black bowl."
    )
    assert "akita_black_bowl_2" not in prompt
    assert "other black bowl" in prompt


def test_target_natural_v1_is_deterministic_and_supported() -> None:
    relations_a = [("akita_black_bowl_1", "is_right_of", "flat_stove_1"), ("akita_black_bowl_1", "is_left_of", "plate_1")]
    relations_b = list(reversed(relations_a))
    assert format_target_natural_v1("Do it.", relations_a) == format_target_natural_v1("Do it.", relations_b)
    assert normalize_context_format(TARGET_NATURAL_FORMAT) == TARGET_NATURAL_FORMAT
    assert format_scene_context("Do it.", relations_a, TARGET_NATURAL_FORMAT) == format_target_natural_v1("Do it.", relations_a)


def test_target_natural_v1_empty_graph_is_explicit() -> None:
    prompt = format_target_natural_v1("Move the bowl.", [])
    assert "has no known spatial relation" in prompt


def test_graph_tokenizer_audit_uses_canonical_contract_and_rejects_truncation() -> None:
    tokenizer = _FakeTokenizer()
    # The fake decoder is intentionally not used for task retention here;
    # this test exercises the exact no-truncation gate.
    report = audit_graph_prompts(["Task: A\nSpatial relations: The target black bowl is left of the plate."], tokenizer=tokenizer)
    assert report["contract"] == GRAPH_TOKENIZER_CONTRACT
    assert report["truncation_checked"] is True
    long_prompt = "x" * (GRAPH_TOKENIZER_CONTRACT["max_length"] + 1)
    try:
        audit_graph_prompts([long_prompt], tokenizer=tokenizer)
    except RuntimeError as exc:
        assert "exceeds 96" in str(exc)
    else:
        raise AssertionError("overlong graph prompt was accepted")


def test_graph_tokenizer_audit_records_vocab_identity_and_accounts_for_serialized_newline() -> None:
    tokenizer = _FakeTokenizer()
    report = audit_graph_prompts(["x" * 95], tokenizer=tokenizer)
    expected_vocab = hashlib.sha256(
        json.dumps(
            sorted((str(token), int(index)) for token, index in tokenizer.get_vocab().items()),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert report["vocab_sha256"] == expected_vocab
    assert report["tokenizer_vocab_sha256"] == expected_vocab

    # LeRobot appends one newline before tokenization; a prompt occupying all
    # 96 available positions must therefore be rejected, not silently cut.
    try:
        audit_graph_prompts(["x" * 96], tokenizer=tokenizer)
    except RuntimeError as exc:
        assert "exceeds 96" in str(exc)
    else:
        raise AssertionError("newline-expanded graph prompt was accepted at the token limit")


def test_graph_tokenizer_audit_requires_a_nonempty_vocabulary() -> None:
    class NoVocabularyTokenizer(_FakeTokenizer):
        def get_vocab(self):
            return {}

    try:
        audit_graph_prompts(["Task: Do it."], tokenizer=NoVocabularyTokenizer())
    except RuntimeError as exc:
        assert "vocabulary" in str(exc)
    else:
        raise AssertionError("tokenizer without vocabulary identity was accepted")


def test_graph_pair_manifest_binds_canonical_extractor_source_and_verifies_drift() -> None:
    """Offline/live parity is only reproducible when the extractor is sealed."""
    converter = Path(__file__).resolve().parents[1] / "hdf5_to_lerobot_dataset.py"
    source = converter.read_text(encoding="utf-8")
    assert '"graph_extractor_sha256": sha256_file(GRAPH_EXTRACTOR_PATH)' in source
    verify_start = source.index("def _load_sealed_manifest")
    verify_end = source.index("def validate_verified_pair", verify_start)
    verifier = source[verify_start:verify_end]
    assert "manifest.get(\"graph_extractor_sha256\")" in verifier
    assert "sha256_file(GRAPH_EXTRACTOR_PATH)" in verifier


def test_target_natural_v1_uses_exact_target_centric_grammar_and_bowl_identities() -> None:
    prompt = format_target_natural_v1(
        "Pick up the target bowl and place it on the plate.",
        [
            ("akita_black_bowl_1", "is_behind", "flat_stove_1"),
            ("akita_black_bowl_1", "is_left_of", "akita_black_bowl_2"),
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
        ],
    )
    assert prompt == (
        "Task: Pick up the target bowl and place it on the plate.\n"
        "Spatial relations: The target black bowl is behind the stove, is left of the other black bowl, "
        "and is left of the plate."
    )
    assert "akita_black_bowl_1" not in prompt
    assert "akita_black_bowl_2" not in prompt
    assert "black bowl is left of the black bowl" not in prompt


def test_target_natural_v1_sorting_and_dedup_are_independent_of_input_order() -> None:
    relations = [
        ("akita_black_bowl_1", "is_right_of", "wooden_cabinet_1"),
        ("akita_black_bowl_1", "is_left_of", "plate_1"),
        ("akita_black_bowl_1", "is_left_of", "plate_1"),
        ("plate_1", "is_left_of", "akita_black_bowl_1"),
        ("akita_black_bowl_1", "is_in_front_of", "cookies_1"),
        ("malformed", "is_left_of"),
    ]
    shuffled = [relations[4], relations[2], relations[5], relations[0], relations[3], relations[1]]
    expected = (
        "Task: Do it.\n"
        "Spatial relations: The target black bowl is in front of the cookie box, is left of the plate, "
        "and is right of the wooden cabinet."
    )
    assert format_target_natural_v1("Do it.", relations) == expected
    assert format_target_natural_v1("Do it.", shuffled) == expected


def test_target_natural_v1_can_disambiguate_a_nondefault_target_subject() -> None:
    prompt = format_target_natural_v1(
        "Move the other bowl.",
        [("akita_black_bowl_2", "is_left_of", "plate_1")],
        target_subject="akita_black_bowl_2",
    )
    assert prompt.endswith("The other black bowl is left of the plate.")
    assert "target black bowl" not in prompt


def _write_policy_snapshot(root: Path, *, max_length: int) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "weights.bin").write_bytes(b"weights")
    (root / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [
                    {"registry_name": "smolvla_new_line_processor", "config": {}},
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {
                            "max_length": max_length,
                            "task_key": "task",
                            "truncation": True,
                            "padding": GRAPH_TOKENIZER_CONTRACT["padding"],
                            "padding_side": GRAPH_TOKENIZER_CONTRACT["padding_side"],
                            "tokenizer_name": GRAPH_TOKENIZER_CONTRACT["model_id"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_graph_preprocessor_rejects_policy_config_96_with_serialized_step_48(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    _write_policy_snapshot(policy, max_length=48)
    try:
        validate_serialized_graph_preprocessor(policy)
    except RuntimeError as exc:
        assert "max_length" in str(exc)
        assert "96" in str(exc)
    else:
        raise AssertionError("graph preprocessor accepted a serialized 48-token step")


def test_graph_policy_snapshot_rewrites_only_graph_tokenizer_budget(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_policy_snapshot(source, max_length=48)
    (source / "tokenizer").mkdir()
    (source / "tokenizer" / "tokenizer.json").write_text("fake", encoding="utf-8")
    tokenizer_files = {"tokenizer.json": hashlib.sha256(b"fake").hexdigest()}
    (source / "tokenizer_provenance.json").write_text(
        json.dumps({
            "materialized": True,
            "model_id": GRAPH_TOKENIZER_CONTRACT["model_id"],
            "revision": GRAPH_TOKENIZER_CONTRACT["revision"],
            "vocab_sha256": "offline-test-vocab",
            "files": tokenizer_files,
            "tree_sha256": hashlib.sha256(
                json.dumps(tokenizer_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }),
        encoding="utf-8",
    )
    # Runtime Hugging Face cache metadata is copied with the snapshot but is
    # intentionally excluded from the authenticated policy-file inventory.
    (source / ".cache" / "huggingface").mkdir(parents=True)
    (source / ".cache" / "huggingface" / "CACHEDIR.TAG").write_text(
        "runtime metadata", encoding="utf-8"
    )
    source_files = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*") if path.is_file() and ".cache" not in path.parts
    }
    (source / "base_snapshot_manifest.json").write_text(
        json.dumps({"revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102", "files": source_files}), encoding="utf-8"
    )
    target = tmp_path / "graph96"
    report = prepare_graph_policy_snapshot(source, target)
    assert report["reused"] is False
    source_steps = json.loads((source / "policy_preprocessor.json").read_text(encoding="utf-8"))["steps"]
    source_tokenizer = next(step for step in source_steps if step["registry_name"] == "tokenizer_processor")
    assert source_tokenizer["config"]["max_length"] == 48
    evidence = validate_serialized_graph_preprocessor(target)
    assert evidence["max_length"] == 96
    assert evidence["tokenizer_name"] == str((target / "tokenizer").resolve())
    assert (target / "base_snapshot_manifest.json").is_file()
    assert not os.path.samefile(source / "tokenizer" / "tokenizer.json", target / "tokenizer" / "tokenizer.json")


def test_graph_policy_snapshot_requires_pinned_revision_and_rejects_file_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_policy_snapshot(source, max_length=48)
    (source / "tokenizer").mkdir()
    (source / "tokenizer" / "tokenizer.json").write_text("fake", encoding="utf-8")
    tokenizer_files = {"tokenizer.json": hashlib.sha256(b"fake").hexdigest()}
    (source / "tokenizer_provenance.json").write_text(
        json.dumps({
            "materialized": True,
            "model_id": GRAPH_TOKENIZER_CONTRACT["model_id"],
            "revision": GRAPH_TOKENIZER_CONTRACT["revision"],
            "vocab_sha256": "offline-test-vocab",
            "files": tokenizer_files,
            "tree_sha256": hashlib.sha256(
                json.dumps(tokenizer_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }),
        encoding="utf-8",
    )
    source_files = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    (source / "base_snapshot_manifest.json").write_text(
        json.dumps({"revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102", "files": source_files}),
        encoding="utf-8",
    )

    target = tmp_path / "graph96"
    prepare_graph_policy_snapshot(source, target)
    evidence = validate_serialized_graph_preprocessor(target, require_snapshot_manifest=True)
    assert evidence["snapshot_manifest"]["revision"] == "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"

    (target / "weights.bin").write_bytes(b"tampered")
    try:
        validate_serialized_graph_preprocessor(target, require_snapshot_manifest=True)
    except RuntimeError as exc:
        assert "file hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered graph policy file was accepted")
