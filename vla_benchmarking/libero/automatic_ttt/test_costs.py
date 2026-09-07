from __future__ import annotations

import pytest

from .costs import CostAccountingError, CostMeter, SensorReading


def test_phase_appends_success_receipt_and_nonnegative_counters():
    meter = CostMeter()
    with meter.phase("trial-0", "collection") as phase:
        phase.increment("env_steps", 3)
        phase.set_sensor("energy_wh", SensorReading.unavailable("NVML not installed"))
    assert len(meter.receipts) == 1
    receipt = meter.receipts[0]
    assert receipt.status == "success"
    assert receipt.sequence == 0
    assert receipt.counters["env_steps"] == 3
    assert receipt.sensors["energy_wh"].value is None


def test_exception_appends_failed_receipt_and_is_not_suppressed():
    meter = CostMeter()
    with pytest.raises(RuntimeError, match="sim failure"):
        with meter.phase("trial-failed", "evaluation") as phase:
            phase.increment("episodes", 1)
            raise RuntimeError("sim failure")
    assert meter.receipts[0].status == "exception"
    assert meter.receipts[0].exception_type == "RuntimeError"
    assert meter.aggregate().failed_receipt_count == 1


def test_negative_values_and_missing_unavailable_reason_are_rejected():
    with pytest.raises(CostAccountingError):
        SensorReading(-1)
    with pytest.raises(CostAccountingError):
        SensorReading.unavailable("")
    meter = CostMeter()
    with pytest.raises(CostAccountingError):
        with meter.phase("t", "training") as phase:
            phase.increment("steps", -1)


def test_aggregate_retains_failed_trials_and_sums_available_sensors():
    meter = CostMeter()
    with meter.phase("ok", "collection") as phase:
        phase.finish(counters={"gpu_hours": 1}, sensors={"energy_wh": SensorReading(2)})
    with pytest.raises(ValueError):
        with meter.phase("failed", "training"):
            raise ValueError("out of memory")
    with meter.phase("ok", "evaluation") as phase:
        phase.finish(counters={"gpu_hours": 0.5}, sensors={"energy_wh": SensorReading(3)})
    aggregate = meter.aggregate()
    assert aggregate.counters["gpu_hours"] == 1.5
    assert aggregate.sensors["energy_wh"].value == 5
    assert {trial.trial_id for trial in aggregate.trials} == {"ok", "failed"}
    assert next(trial for trial in aggregate.trials if trial.trial_id == "failed").failed_phase_count == 1


def test_aggregate_keeps_sensor_name_when_every_measurement_is_unavailable():
    meter = CostMeter()
    with meter.phase("trial", "collection") as phase:
        phase.set_sensor("gpu_energy_wh", SensorReading.unavailable("energy counter disabled"))
    aggregate = meter.aggregate()
    assert aggregate.sensors["gpu_energy_wh"].value is None
    assert aggregate.sensors["gpu_energy_wh"].unavailable_reason == "energy counter disabled"


def test_receipts_are_append_only_snapshot():
    meter = CostMeter()
    with meter.phase("t", "p"):
        pass
    snapshot = meter.receipts
    with meter.phase("t", "q"):
        pass
    assert len(snapshot) == 1
    assert len(meter.receipts) == 2
