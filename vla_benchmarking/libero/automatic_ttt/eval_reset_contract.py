"""Immutable reset identities for sealed automatic-TTT evaluation.

The evaluator and the Arrow collector need to agree on *which* randomized
LIBERO initial states were used.  A numeric seed alone is not enough because
LIBERO seeds can map to the same finite init-state table.  This module keeps
that agreement in a small, dependency-light contract:

* the contract always describes one task;
* seeds are exactly ``1000..1009`` and indices exactly ``0..9``;
* every hash comes from an instantiated environment's reset diagnostics; and
* the resulting JSON file is write-once and atomically published.

The probe API accepts an environment factory, so launchers can reserve the
states without constructing or calling a VLA.  ``probe_libero_eval_resets`` is
the convenience adapter for the repository's canonical LIBERO builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import ContractError


RESET_CONTRACT_SCHEMA = "automatic_ttt.eval_reset_contract.v1"
SEALED_EVAL_SEEDS: tuple[int, ...] = tuple(range(1000, 1010))
SEALED_EVAL_INIT_STATE_INDICES: tuple[int, ...] = tuple(range(10))
_HEX = frozenset("0123456789abcdefABCDEF")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value


def _audit_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "randomization_audit.jsonl"
    if not path.is_file():
        raise ContractError(f"randomization audit is missing: {path}")
    return path


def _sha256_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise ContractError(f"{label} must be a 64-character SHA-256 digest")
    return value.lower()


def _identity_from_diagnostics(diagnostics: Mapping[str, Any], *, task_id: int, seed: int, expected_index: int | None = None) -> dict[str, Any]:
    """Normalize one actual reset diagnostic into a contract reservation."""
    observed_task = diagnostics.get("task_id", task_id)
    if isinstance(observed_task, bool) or not isinstance(observed_task, int) or int(observed_task) != int(task_id):
        raise ContractError("reset diagnostics task_id differs from the requested task")
    selected = diagnostics.get(
        "selected_init_state_index",
        diagnostics.get("init_state_index", diagnostics.get("selected_index")),
    )
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
        raise ContractError("reset diagnostics selected init-state index is malformed")
    if expected_index is not None and int(selected) != int(expected_index):
        raise ContractError(
            f"reset diagnostics selected index differs from reservation: expected {expected_index}, got {selected}"
        )
    digest = diagnostics.get("init_state_sha256", diagnostics.get("selected_row_sha256"))
    return {
        "seed": int(seed),
        "init_state_index": int(selected),
        "init_state_sha256": _sha256_digest(digest, "reset diagnostics init-state hash"),
    }


def _validate_contract_payload(payload: Mapping[str, Any], *, task_id: int | None = None) -> dict[str, Any]:
    if payload.get("schema") != RESET_CONTRACT_SCHEMA or payload.get("schema_version") != 1:
        raise ContractError(f"reset contract schema must be {RESET_CONTRACT_SCHEMA}")
    value_task = payload.get("task_id")
    if isinstance(value_task, bool) or not isinstance(value_task, int) or not 0 <= value_task <= 9:
        raise ContractError("reset contract task_id must be an integer in 0..9")
    if task_id is not None and int(value_task) != int(task_id):
        raise ContractError("reset contract task_id differs from requested task")

    seeds = payload.get("eval_seeds")
    indices = payload.get("reserved_eval_init_state_indices")
    hashes = payload.get("reserved_eval_init_state_hashes")
    if list(seeds or ()) != list(SEALED_EVAL_SEEDS):
        raise ContractError("reset contract eval_seeds must be exactly 1000..1009")
    if list(indices or ()) != list(SEALED_EVAL_INIT_STATE_INDICES):
        raise ContractError("reset contract init-state indices must be exactly 0..9")
    if not isinstance(hashes, list) or len(hashes) != 10:
        raise ContractError("reset contract must contain exactly ten init-state hashes")
    normalized_hashes = [_sha256_digest(item, "reset contract init-state hash") for item in hashes]
    if len(set(normalized_hashes)) != 10:
        raise ContractError("reset contract init-state hashes must be unique")

    reservations = payload.get("reservations")
    if not isinstance(reservations, list) or len(reservations) != 10:
        raise ContractError("reset contract must contain exactly ten reservations")
    expected_reservations = [
        {"seed": seed, "init_state_index": index, "init_state_sha256": digest}
        for seed, index, digest in zip(SEALED_EVAL_SEEDS, SEALED_EVAL_INIT_STATE_INDICES, normalized_hashes)
    ]
    normalized_reservations: list[dict[str, Any]] = []
    for item, expected in zip(reservations, expected_reservations):
        if not isinstance(item, Mapping):
            raise ContractError("reset contract reservation must be an object")
        try:
            seed = int(item["seed"])
            index = int(item["init_state_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("reset contract reservation identity is malformed") from exc
        digest = _sha256_digest(item.get("init_state_sha256"), "reset contract reservation hash")
        normalized = {"seed": seed, "init_state_index": index, "init_state_sha256": digest}
        if normalized != expected:
            raise ContractError("reset contract reservation does not match sealed seed/index/hash arrays")
        normalized_reservations.append(normalized)

    result = dict(payload)
    result["schema"] = RESET_CONTRACT_SCHEMA
    result["schema_version"] = 1
    result["task_id"] = int(value_task)
    result["eval_seeds"] = list(SEALED_EVAL_SEEDS)
    result["reserved_eval_init_state_indices"] = list(SEALED_EVAL_INIT_STATE_INDICES)
    result["reserved_eval_init_state_hashes"] = normalized_hashes
    result["reservations"] = normalized_reservations
    return result


def make_eval_reset_contract(
    task_id: int,
    diagnostics: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate a contract from ten actual reset diagnostics.

    Each item must contain ``seed``, ``selected_init_state_index`` (or
    ``selected_index``), and ``init_state_sha256`` (or
    ``selected_row_sha256``).  The input order is not trusted; the sealed seed
    and index sets are checked and the output is sorted by evaluation seed.
    """
    task_id = int(task_id)
    if not 0 <= task_id <= 9:
        raise ContractError("task_id must be an integer in 0..9")
    rows = list(diagnostics)
    if len(rows) != 10:
        raise ContractError("reset diagnostics must contain exactly ten rows")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContractError("reset diagnostics row must be an object")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEALED_EVAL_SEEDS:
            raise ContractError("reset diagnostics seed must be one of 1000..1009")
        normalized.append(_identity_from_diagnostics(row, task_id=task_id, seed=seed))
    normalized.sort(key=lambda item: item["seed"])
    if [row["seed"] for row in normalized] != list(SEALED_EVAL_SEEDS):
        raise ContractError("reset diagnostics seeds must be unique and exactly 1000..1009")
    if [row["init_state_index"] for row in normalized] != list(SEALED_EVAL_INIT_STATE_INDICES):
        raise ContractError("reset diagnostics indices must be unique and exactly 0..9 in seed order")
    hashes = [row["init_state_sha256"] for row in normalized]
    if len(set(hashes)) != 10:
        raise ContractError("reset diagnostics init-state hashes must be unique")
    payload = {
        "schema": RESET_CONTRACT_SCHEMA,
        "schema_version": 1,
        "task_id": task_id,
        "source": "actual_reset_diagnostics",
        "hash_algorithm": "sha256",
        "eval_seeds": list(SEALED_EVAL_SEEDS),
        "reserved_eval_init_state_indices": list(SEALED_EVAL_INIT_STATE_INDICES),
        "reserved_eval_init_state_hashes": hashes,
        "reservations": normalized,
    }
    return _validate_contract_payload(payload, task_id=task_id)


def write_eval_reset_contract(contract: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically publish a validated, immutable contract JSON file."""
    payload = _validate_contract_payload(contract, task_id=int(contract.get("task_id", -1)))
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable reset contract: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"stale temporary reset contract exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # link() is the publish operation: unlike replace(), it cannot clobber
        # a contract created concurrently by another process.
        os.link(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    try:
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return destination


def load_eval_reset_contract(path: str | Path, *, task_id: int | None = None) -> dict[str, Any]:
    """Read and validate a previously published reset contract."""
    return _validate_contract_payload(_read_json(Path(path), "reset contract"), task_id=task_id)


def probe_eval_reset_contract(
    task_id: int,
    environment_factory: Callable[..., Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Probe all sealed resets without VLA actions and optionally publish.

    ``environment_factory`` must accept keyword arguments
    ``task_id``, ``seed``, and ``init_state_index`` and return an already
    instantiated/reset environment exposing ``_arrow_init_state_diagnostics``
    (or the equivalent evidence accepted by ``init_state_evidence``).  Every
    returned environment is closed in a ``finally`` block, including when
    diagnostics validation fails.
    """
    rows: list[dict[str, Any]] = []
    for seed, index in zip(SEALED_EVAL_SEEDS, SEALED_EVAL_INIT_STATE_INDICES):
        environment = None
        active_error: BaseException | None = None
        try:
            environment = environment_factory(task_id=task_id, seed=seed, init_state_index=index)
            diagnostics = getattr(environment, "_arrow_init_state_diagnostics", None)
            if not isinstance(diagnostics, Mapping):
                try:
                    from vla_benchmarking.libero.evaluation.randomize_scenes import init_state_evidence
                    diagnostics = init_state_evidence(environment)
                except Exception as exc:
                    raise ContractError("environment lacks actual reset diagnostics") from exc
            row = dict(diagnostics)
            row["task_id"] = int(task_id)
            row["seed"] = int(seed)
            rows.append(_identity_from_diagnostics(row, task_id=int(task_id), seed=seed, expected_index=index))
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            if environment is not None:
                close = getattr(environment, "close", None)
                if not callable(close):
                    if active_error is None:
                        raise ContractError("probed environment does not expose close()")
                else:
                    try:
                        close()
                    except BaseException:
                        if active_error is None:
                            raise
    contract = make_eval_reset_contract(int(task_id), rows)
    if output_path is not None:
        write_eval_reset_contract(contract, output_path)
    return contract


def probe_libero_eval_resets(
    task_id: int,
    output_path: str | Path | None = None,
    *,
    resolution: int = 256,
) -> dict[str, Any]:
    """Probe the canonical LIBERO builder without loading or calling a VLA."""
    from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import build_libero_env

    def factory(*, task_id: int, seed: int, init_state_index: int) -> Any:
        return build_libero_env(
            int(task_id), int(seed), int(resolution),
            suite_mode="sealed_randomized",
            extra_camera_names=("robot0_eye_in_hand",),
            init_state_index=int(init_state_index),
        )

    return probe_eval_reset_contract(int(task_id), factory, output_path=output_path)


def _audit_rows(path: str | Path) -> list[dict[str, Any]]:
    audit = _audit_path(path)
    rows: list[dict[str, Any]] = []
    try:
        lines = audit.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"randomization audit is unreadable: {audit}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"malformed randomization audit line {line_number}: {audit}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"randomization audit line {line_number} is not an object")
        rows.append(value)
    return rows


def contract_from_randomization_audit(
    task_id: int,
    audit_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a contract from an audit produced by the actual evaluator."""
    rows = _audit_rows(audit_path)
    identities: list[dict[str, Any]] = []
    for row in rows:
        try:
            row_task = int(row.get("task_id", -1))
        except (TypeError, ValueError) as exc:
            raise ContractError("audit task_id is malformed") from exc
        if row_task != int(task_id):
            continue
        details = row.get("details")
        identity = details.get("reset_identity") if isinstance(details, Mapping) else None
        if not isinstance(identity, Mapping):
            continue
        try:
            sequence = int(identity.get("reset_sequence", row.get("reset_sequence", 0)))
        except (TypeError, ValueError) as exc:
            raise ContractError("audit reset identity sequence is malformed") from exc
        if not 1 <= sequence <= 10:
            raise ContractError("audit reset identity sequence must be 1..10")
        item = dict(identity)
        item["task_id"] = int(task_id)
        item["seed"] = 999 + sequence
        identities.append(item)
    if len(identities) != 10:
        raise ContractError("audit must contain exactly ten reset identities")
    contract = make_eval_reset_contract(int(task_id), identities)
    if output_path is not None:
        write_eval_reset_contract(contract, output_path)
    return contract


def validate_adapted_randomization_audit(
    contract: Mapping[str, Any] | str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    """Verify adapted evaluation reset identities against a reservation."""
    reservation = load_eval_reset_contract(contract) if isinstance(contract, (str, Path)) else _validate_contract_payload(contract)
    expected_by_sequence = {index + 1: item for index, item in enumerate(reservation["reservations"])}
    rows = _audit_rows(audit_path)
    observed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if int(row.get("task_id", -1)) != reservation["task_id"]:
            continue
        details = row.get("details")
        identity = details.get("reset_identity") if isinstance(details, Mapping) else None
        if not isinstance(identity, Mapping):
            raise ContractError("adapted randomization audit row lacks reset_identity")
        try:
            sequence = int(identity.get("reset_sequence", row.get("reset_sequence", 0)))
            env_index = int(identity.get("env_index", row.get("env_index", -1)))
            selected = int(identity.get("selected_init_state_index"))
        except (TypeError, ValueError) as exc:
            raise ContractError("adapted randomization audit reset identity is malformed") from exc
        if "reset_sequence" in row and int(row["reset_sequence"]) != sequence:
            raise ContractError("adapted randomization audit row and identity sequences differ")
        if "env_index" in row and int(row["env_index"]) != env_index:
            raise ContractError("adapted randomization audit row and identity env indices differ")
        if sequence not in expected_by_sequence or env_index != 0:
            raise ContractError("adapted randomization audit reset sequence/env index is invalid")
        if sequence in observed:
            raise ContractError("adapted randomization audit contains duplicate reset identity")
        expected = expected_by_sequence[sequence]
        digest = _sha256_digest(identity.get("init_state_sha256"), "adapted audit init-state hash")
        if selected != expected["init_state_index"] or digest != expected["init_state_sha256"]:
            raise ContractError("adapted evaluation reset identity differs from reserved identity")
        if "seed" in row:
            try:
                observed_seed = int(row["seed"])
            except (TypeError, ValueError) as exc:
                raise ContractError("adapted evaluation audit seed is malformed") from exc
            if observed_seed != expected["seed"]:
                raise ContractError("adapted evaluation audit seed differs from reserved seed")
        if row.get("status") not in (None, "ok"):
            raise ContractError("adapted randomization audit contains a non-ok reset")
        observed[sequence] = {
            "seed": expected["seed"],
            "init_state_index": selected,
            "init_state_sha256": digest,
        }
    if set(observed) != set(expected_by_sequence):
        raise ContractError("adapted randomization audit must contain all ten reserved resets")
    return {
        "schema": RESET_CONTRACT_SCHEMA,
        "task_id": reservation["task_id"],
        "validated": True,
        "episodes": 10,
        "seeds": [observed[index]["seed"] for index in range(1, 11)],
        "init_state_indices": [observed[index]["init_state_index"] for index in range(1, 11)],
        "hashes": [observed[index]["init_state_sha256"] for index in range(1, 11)],
    }


# ``create`` is the launcher-facing spelling; keep ``make`` as the explicit
# pure-construction API used by tests and library callers.
create_eval_reset_contract = make_eval_reset_contract


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create/validate sealed automatic-TTT reset contracts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe", help="probe canonical LIBERO resets")
    probe.add_argument("--task-id", type=int, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--resolution", type=int, default=256)
    from_audit = subparsers.add_parser("from-audit", help="create contract from evaluator reset audit")
    from_audit.add_argument("--task-id", type=int, required=True)
    from_audit.add_argument("--audit", type=Path, required=True)
    from_audit.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-audit", help="validate adapted audit against contract")
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        probe_libero_eval_resets(args.task_id, args.output, resolution=args.resolution)
        return 0
    if args.command == "from-audit":
        contract_from_randomization_audit(args.task_id, args.audit, output_path=args.output)
        return 0
    report = validate_adapted_randomization_audit(args.contract, args.audit)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI exercised by shell launchers
    try:
        raise SystemExit(_main())
    except ContractError as exc:
        print(f"reset-contract: {exc}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "RESET_CONTRACT_SCHEMA",
    "SEALED_EVAL_SEEDS",
    "SEALED_EVAL_INIT_STATE_INDICES",
    "make_eval_reset_contract",
    "create_eval_reset_contract",
    "write_eval_reset_contract",
    "load_eval_reset_contract",
    "probe_eval_reset_contract",
    "probe_libero_eval_resets",
    "contract_from_randomization_audit",
    "validate_adapted_randomization_audit",
]
