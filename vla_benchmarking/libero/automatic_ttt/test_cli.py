import json
from argparse import Namespace

from .cli import _cmd_collect, _cmd_evaluate, _cmd_report, _cmd_train
from .config import config_from_dict, save_config
from .evaluation import TrialMetric, write_metrics


def _config(tmp_path):
    config = config_from_dict(
        {
            "fidelity_mode": "algorithmic_port",
            "vla_names": ["openvla"],
            "task_ids": [0],
            "episodes_per_task": 1,
            "controls": ["frozen_baseline", "hybrid"],
            "split": {"eval_episode_ids": ["task00_seed10_eval00"]},
            "controls": ["frozen_baseline", "adapted", "hybrid"],
        }
    )
    path = tmp_path / "config.json"
    save_config(config, path)
    return path


def _metric(condition):
    return TrialMetric(
        policy_id="openvla-policy",
        vla="openvla",
        task_id=0,
        seed=10,
        condition=condition,
        success=condition != "frozen_baseline",
        teacher_used=condition == "hybrid",
        teacher_success=condition == "hybrid",
        episode_id="task00_seed10_eval00",
        initial_state_hash="state-hash",
        checkpoint_lineage="checkpoint-lineage",
        metadata={"teacher_eligible": True},
    )


def injected_factory(*, config, args):
    """Test-only host callback proving the CLI injection path is executable."""
    return {"status": "EXECUTED", "operation": args.command, "config_digest": config.digest()}


def test_report_empty_is_incomplete(tmp_path):
    code = _cmd_report(Namespace(run_root=str(tmp_path), config=None))
    assert code != 0


def test_report_uses_strict_paired_report(tmp_path, capsys):
    config_path = _config(tmp_path)
    run_root = tmp_path / "run"
    write_metrics([_metric("frozen_baseline"), _metric("adapted"), _metric("hybrid")], run_root / "metrics.json")
    code = _cmd_report(Namespace(run_root=str(run_root), config=str(config_path)))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completeness_receipt"]["status"] == "COMPLETE"
    row = payload["paired_report"][0]
    assert row["paired_trials"] == 1
    assert "baseline_wilson95" in row
    assert "teacher_recovery_rate" in row
    assert "hybrid_success_rate" in row


def test_report_rejects_unpaired_conditions(tmp_path):
    config_path = _config(tmp_path)
    run_root = tmp_path / "run"
    write_metrics([_metric("frozen_baseline")], run_root / "metrics.json")
    code = _cmd_report(Namespace(run_root=str(run_root), config=str(config_path)))
    assert code != 0


def test_runtime_factory_executes_each_operation(tmp_path):
    config_path = _config(tmp_path)
    factory = "vla_benchmarking.libero.automatic_ttt.test_cli:injected_factory"
    for command, callback in (("collect", _cmd_collect), ("train", _cmd_train), ("evaluate", _cmd_evaluate)):
        code = callback(
            Namespace(
                command=command,
                config=str(config_path),
                factory=factory,
                dry_run=False,
            )
        )
        assert code == 0


def test_report_rejects_textual_boolean_flags(tmp_path):
    config_path = _config(tmp_path)
    run_root = tmp_path / "run"
    record = _metric("frozen_baseline").to_json()
    record["teacher_used"] = "false"
    (run_root).mkdir()
    (run_root / "metrics.json").write_text(json.dumps({"episodes": [record]}), encoding="utf-8")
    code = _cmd_report(Namespace(run_root=str(run_root), config=str(config_path)))
    assert code != 0
