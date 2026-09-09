import json
from pathlib import Path

import pytest

from .contracts import ContractError
from .eval_reset_contract import (
    RESET_CONTRACT_SCHEMA,
    contract_from_randomization_audit,
    load_eval_reset_contract,
    make_eval_reset_contract,
    probe_eval_reset_contract,
    validate_adapted_randomization_audit,
    write_eval_reset_contract,
)


def _diagnostics(*, task_id=0, duplicate_hash=False):
    rows = []
    for seed in range(1000, 1010):
        index = seed - 1000
        rows.append({
            "task_id": task_id,
            "seed": seed,
            "selected_init_state_index": index,
            "selected_row_sha256": ("a" * 63 + "b") if duplicate_hash else f"{seed:064x}",
        })
    return rows


def _audit_rows(task_id=0, hashes=None):
    hashes = hashes or [f"{seed:064x}" for seed in range(1000, 1010)]
    return [
        {
            "task_id": task_id,
            "env_index": 0,
            "reset_sequence": sequence,
            "status": "ok",
            "details": {
                "reset_identity": {
                    "task_id": task_id,
                    "env_index": 0,
                    "reset_sequence": sequence,
                    "selected_init_state_index": sequence - 1,
                    "init_state_sha256": hashes[sequence - 1],
                }
            },
        }
        for sequence in range(1, 11)
    ]


def test_make_and_load_contract_has_sealed_reservations(tmp_path: Path):
    contract = make_eval_reset_contract(0, _diagnostics())
    assert contract["schema"] == RESET_CONTRACT_SCHEMA
    assert contract["eval_seeds"] == list(range(1000, 1010))
    assert contract["reserved_eval_init_state_indices"] == list(range(10))
    assert [item["seed"] for item in contract["reservations"]] == list(range(1000, 1010))

    path = tmp_path / "eval_reset_contract.json"
    write_eval_reset_contract(contract, path)
    assert load_eval_reset_contract(path, task_id=0) == contract
    with pytest.raises(FileExistsError):
        write_eval_reset_contract(contract, path)


def test_duplicate_hashes_are_rejected():
    with pytest.raises(ContractError, match="hashes must be unique"):
        make_eval_reset_contract(0, _diagnostics(duplicate_hash=True))


def test_wrong_seed_or_index_is_rejected():
    rows = _diagnostics()
    rows[0]["seed"] = 1010
    with pytest.raises(ContractError, match="seed"):
        make_eval_reset_contract(0, rows)

    rows = _diagnostics()
    rows[0]["selected_init_state_index"] = 1
    with pytest.raises(ContractError, match="indices"):
        make_eval_reset_contract(0, rows)


def test_probe_closes_environment_when_diagnostics_fail():
    class FakeEnvironment:
        def __init__(self):
            self.closed = False
            self._arrow_init_state_diagnostics = {"selected_index": 99, "selected_row_sha256": "a" * 64}

        def close(self):
            self.closed = True

    environments = []

    def factory(*, task_id, seed, init_state_index):
        environment = FakeEnvironment()
        environments.append(environment)
        return environment

    with pytest.raises(ContractError, match="differs from reservation"):
        probe_eval_reset_contract(0, factory)
    assert len(environments) == 1
    assert environments[0].closed is True


def test_probe_closes_all_environments_and_writes_contract(tmp_path: Path):
    environments = []

    class FakeEnvironment:
        def __init__(self, index, digest):
            self.closed = False
            self._arrow_init_state_diagnostics = {
                "selected_index": index,
                "selected_row_sha256": digest,
            }

        def close(self):
            self.closed = True

    def factory(*, task_id, seed, init_state_index):
        environment = FakeEnvironment(init_state_index, f"{seed:064x}")
        environments.append(environment)
        return environment

    output = tmp_path / "contract.json"
    contract = probe_eval_reset_contract(0, factory, output_path=output)
    assert output.is_file()
    assert len(environments) == 10
    assert all(environment.closed for environment in environments)
    assert load_eval_reset_contract(output) == contract


def test_contract_from_audit_and_adapted_audit_validation(tmp_path: Path):
    audit = tmp_path / "randomization_audit.jsonl"
    audit.write_text("".join(json.dumps(row) + "\n" for row in _audit_rows()), encoding="utf-8")
    output = tmp_path / "contract.json"
    contract = contract_from_randomization_audit(0, audit, output_path=output)
    assert contract["eval_seeds"] == list(range(1000, 1010))
    report = validate_adapted_randomization_audit(output, audit)
    assert report["validated"] is True
    assert report["seeds"] == list(range(1000, 1010))


def test_adapted_audit_hash_mismatch_is_rejected(tmp_path: Path):
    contract = make_eval_reset_contract(0, _diagnostics())
    contract_path = tmp_path / "contract.json"
    write_eval_reset_contract(contract, contract_path)
    audit = tmp_path / "randomization_audit.jsonl"
    rows = _audit_rows()
    rows[4]["details"]["reset_identity"]["init_state_sha256"] = "f" * 64
    audit.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ContractError, match="differs from reserved"):
        validate_adapted_randomization_audit(contract_path, audit)


def test_audit_duplicate_reset_is_rejected(tmp_path: Path):
    contract = make_eval_reset_contract(0, _diagnostics())
    contract_path = tmp_path / "contract.json"
    write_eval_reset_contract(contract, contract_path)
    audit = tmp_path / "randomization_audit.jsonl"
    rows = _audit_rows()
    rows[-1]["reset_sequence"] = 1
    rows[-1]["details"]["reset_identity"]["reset_sequence"] = 1
    audit.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate"):
        validate_adapted_randomization_audit(contract_path, audit)
