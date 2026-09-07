"""Append-only, failure-preserving cost and runtime accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Any, Mapping


class CostAccountingError(ValueError):
    pass


def _nonnegative(name: str, value: float | int) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise CostAccountingError(f"{name} must be a finite non-negative number")
    return value


@dataclass(frozen=True)
class SensorReading:
    """A measurement or an explicit unavailable value with a reason."""

    value: float | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is None:
            if not self.unavailable_reason:
                raise CostAccountingError("unavailable sensors require a reason")
        else:
            _nonnegative("sensor value", self.value)
            if self.unavailable_reason is not None:
                raise CostAccountingError("available sensors cannot carry unavailable_reason")

    @classmethod
    def unavailable(cls, reason: str) -> "SensorReading":
        return cls(None, reason)


@dataclass(frozen=True)
class PhaseReceipt:
    trial_id: str
    phase: str
    sequence: int
    status: str
    counters: Mapping[str, float] = field(default_factory=dict)
    sensors: Mapping[str, SensorReading] = field(default_factory=dict)
    exception_type: str | None = None
    exception_message: str | None = None

    def __post_init__(self) -> None:
        if not self.trial_id or not self.phase or self.sequence < 0:
            raise CostAccountingError("receipt requires trial_id, phase, and non-negative sequence")
        if self.status not in {"success", "exception"}:
            raise CostAccountingError("phase status must be success or exception")
        if self.status == "exception" and not self.exception_type:
            raise CostAccountingError("exception receipts require exception_type")
        if self.status == "success" and self.exception_type is not None:
            raise CostAccountingError("success receipts cannot carry exception_type")
        for name, value in self.counters.items():
            if not name:
                raise CostAccountingError("counter names cannot be empty")
            _nonnegative(f"counter {name}", value)
        for name, reading in self.sensors.items():
            if not isinstance(reading, SensorReading):
                raise CostAccountingError(f"sensor {name} must be SensorReading")


@dataclass(frozen=True)
class TrialCostAggregate:
    trial_id: str
    receipt_count: int
    failed_phase_count: int
    counters: Mapping[str, float]
    sensors: Mapping[str, SensorReading]


@dataclass(frozen=True)
class CostAggregate:
    receipt_count: int
    success_receipt_count: int
    failed_receipt_count: int
    counters: Mapping[str, float]
    sensors: Mapping[str, SensorReading]
    trials: tuple[TrialCostAggregate, ...]


class _PhaseScope:
    def __init__(self, meter: "CostMeter", trial_id: str, phase: str) -> None:
        self._meter = meter
        self._trial_id = trial_id
        self._phase = phase
        self._counters: dict[str, float] = {}
        self._sensors: dict[str, SensorReading] = {}
        self._closed = False

    def increment(self, name: str, amount: float = 1) -> None:
        if self._closed:
            raise CostAccountingError("phase is already closed")
        if not name:
            raise CostAccountingError("counter names cannot be empty")
        _nonnegative("counter increment", amount)
        self._counters[name] = self._counters.get(name, 0.0) + float(amount)

    def set_sensor(self, name: str, reading: SensorReading) -> None:
        if self._closed:
            raise CostAccountingError("phase is already closed")
        if not isinstance(reading, SensorReading):
            raise CostAccountingError("set_sensor requires SensorReading")
        self._sensors[name] = reading

    def finish(self, *, counters: Mapping[str, float] | None = None, sensors: Mapping[str, SensorReading] | None = None) -> PhaseReceipt:
        if self._closed:
            raise CostAccountingError("phase is already closed")
        if counters is not None:
            for name, value in counters.items():
                _nonnegative(f"counter {name}", value)
            self._counters.update({name: float(value) for name, value in counters.items()})
        if sensors is not None:
            for name, reading in sensors.items():
                if not isinstance(reading, SensorReading):
                    raise CostAccountingError("sensors must contain SensorReading values")
            self._sensors.update(sensors)
        self._closed = True
        # ``sequence`` is assigned atomically by the meter.  Zero is a valid
        # construction placeholder; callers never observe this unpublished row.
        return self._meter._append(PhaseReceipt(self._trial_id, self._phase, 0, "success", dict(self._counters), dict(self._sensors)))

    def fail(self, exc: BaseException) -> PhaseReceipt:
        if self._closed:
            raise CostAccountingError("phase is already closed")
        self._closed = True
        return self._meter._append(PhaseReceipt(self._trial_id, self._phase, 0, "exception", dict(self._counters), dict(self._sensors), type(exc).__name__, str(exc)))

    def __enter__(self) -> "_PhaseScope":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, _traceback: Any) -> bool:
        if not self._closed:
            if exc is None:
                self.finish()
            else:
                self.fail(exc)
        return False


class CostMeter:
    """Thread-safe append-only phase receipt ledger."""

    def __init__(self) -> None:
        self._receipts: list[PhaseReceipt] = []
        self._lock = threading.Lock()

    @property
    def receipts(self) -> tuple[PhaseReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def phase(self, trial_id: str, phase: str) -> _PhaseScope:
        if not trial_id or not phase:
            raise CostAccountingError("trial_id and phase are required")
        return _PhaseScope(self, trial_id, phase)

    def _append(self, receipt: PhaseReceipt) -> PhaseReceipt:
        with self._lock:
            sequence = len(self._receipts)
            published = PhaseReceipt(receipt.trial_id, receipt.phase, sequence, receipt.status, receipt.counters, receipt.sensors, receipt.exception_type, receipt.exception_message)
            self._receipts.append(published)
            return published

    def aggregate(self) -> CostAggregate:
        receipts = self.receipts
        totals: dict[str, float] = {}
        sensor_values: dict[str, list[float]] = {}
        unavailable_sensors: dict[str, str] = {}
        by_trial: dict[str, list[PhaseReceipt]] = {}
        for receipt in receipts:
            by_trial.setdefault(receipt.trial_id, []).append(receipt)
            for name, value in receipt.counters.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            for name, reading in receipt.sensors.items():
                if reading.value is not None:
                    sensor_values.setdefault(name, []).append(float(reading.value))
                else:
                    unavailable_sensors.setdefault(name, reading.unavailable_reason or "sensor unavailable")
        aggregate_sensors = {}
        for name in set(sensor_values) | set(unavailable_sensors):
            if name in sensor_values and name in unavailable_sensors:
                aggregate_sensors[name] = SensorReading.unavailable("some readings unavailable")
            elif name in sensor_values:
                aggregate_sensors[name] = SensorReading(sum(sensor_values[name]))
            else:
                aggregate_sensors[name] = SensorReading.unavailable(unavailable_sensors[name])
        # Preserve every trial, including those whose only receipt is an exception.
        trials = []
        for trial_id, trial_receipts in by_trial.items():
            trial_totals: dict[str, float] = {}
            trial_sensor_values: dict[str, list[float]] = {}
            trial_unavailable: dict[str, str] = {}
            for receipt in trial_receipts:
                for name, value in receipt.counters.items():
                    trial_totals[name] = trial_totals.get(name, 0.0) + float(value)
                for name, reading in receipt.sensors.items():
                    if reading.value is not None:
                        trial_sensor_values.setdefault(name, []).append(float(reading.value))
                    else:
                        trial_unavailable.setdefault(name, reading.unavailable_reason or "sensor unavailable")
            trial_sensors = {}
            for name in set(trial_sensor_values) | set(trial_unavailable):
                if name in trial_sensor_values and name in trial_unavailable:
                    trial_sensors[name] = SensorReading.unavailable("some readings unavailable")
                elif name in trial_sensor_values:
                    trial_sensors[name] = SensorReading(sum(trial_sensor_values[name]))
                else:
                    trial_sensors[name] = SensorReading.unavailable(trial_unavailable[name])
            trials.append(TrialCostAggregate(trial_id, len(trial_receipts), sum(r.status == "exception" for r in trial_receipts), trial_totals, trial_sensors))
        return CostAggregate(len(receipts), sum(r.status == "success" for r in receipts), sum(r.status == "exception" for r in receipts), totals, aggregate_sensors, tuple(trials))


__all__ = ["CostAccountingError", "CostAggregate", "CostMeter", "PhaseReceipt", "SensorReading", "TrialCostAggregate"]
