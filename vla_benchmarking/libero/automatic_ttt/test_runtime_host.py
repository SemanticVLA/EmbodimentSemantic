from __future__ import annotations

import pytest

from .contracts import EpisodeSpec
from .runtime_host import (
    EnvironmentIdentity,
    OperationResult,
    OperationStatus,
    RuntimeContractError,
    RuntimeHost,
    RuntimeUnavailableError,
)


class FakeFactory:
    factory_id = "fake-libero"
    factory_version = "1"

    def __init__(self):
        self.env = None
        self.reset_calls = 0
        self.close_calls = 0

    def create_environment(self, _episode):
        self.env = {"t": 0}
        return self.env

    def reset_environment(self, environment, _episode):
        self.reset_calls += 1
        environment["t"] = 0
        return {"state": [0]}

    def observe_environment(self, environment):
        return {"state": [environment["t"]]}

    def step_environment(self, environment, _action):
        environment["t"] += 1
        return {"done": environment["t"] >= 1}

    def simulator_state_digest(self, environment):
        return "a" * 64

    def replay_key(self, _environment, episode):
        return f"replay:{episode.task_id}:{episode.seed}"

    def close_environment(self, _environment):
        self.close_calls += 1


def spec():
    return EpisodeSpec("runtime-0", 0, 17, "pick", "openvla")


def test_host_preflight_owns_exactly_one_reset_and_close_and_separates_hashes():
    factory = FakeFactory()
    receipt = RuntimeHost(factory).preflight(spec())
    assert factory.reset_calls == 1
    assert factory.close_calls == 1
    assert receipt.identity.observation_hash != receipt.identity.simulator_state_digest
    assert receipt.identity.replay_key == "replay:0:17"


def test_lease_view_has_no_reset_or_close_and_host_closes_once():
    factory = FakeFactory()
    host = RuntimeHost(factory)
    lease = host.open(spec())
    assert not hasattr(lease.view, "reset")
    assert not hasattr(lease.view, "close")
    lease.close()
    with pytest.raises(Exception, match="more than once"):
        lease.close()
    assert factory.reset_calls == 1 and factory.close_calls == 1


def test_operation_requires_typed_result_and_preserves_exceptions():
    factory = FakeFactory()
    host = RuntimeHost(factory)
    lease = host.open(spec())
    with pytest.raises(RuntimeContractError, match="OperationResult"):
        host.execute(lease, "bad", lambda _view: {"success": True})
    result = host.execute(lease, "boom", lambda _view: (_ for _ in ()).throw(ValueError("bad")))
    assert result.status is OperationStatus.ERROR
    assert result.error_type == "ValueError"
    lease.close()


def test_success_result_must_bind_to_the_immutable_lease_identity():
    factory = FakeFactory()
    host = RuntimeHost(factory)
    lease = host.open(spec())
    with pytest.raises(RuntimeContractError, match="exact lease environment identity"):
        host.execute(lease, "missing-identity", lambda _view: OperationResult.success("missing-identity", object()))
    result = host.execute(
        lease,
        "typed",
        lambda view: OperationResult.success("typed", (True,), environment_identity=view.identity),
    )
    assert result.status is OperationStatus.SUCCESS
    lease.close()


def test_success_factory_rejects_mapping_value():
    with pytest.raises(RuntimeContractError, match="typed object"):
        OperationResult.success("x", {"success": True})


def test_missing_factory_fails_closed():
    with pytest.raises(RuntimeUnavailableError):
        RuntimeHost(None)


def test_host_blocks_before_first_action_when_live_first_observation_differs():
    class MismatchedFactory(FakeFactory):
        def observe_environment(self, environment):
            return {"state": [environment["t"] + 1]}

    factory = MismatchedFactory()
    host = RuntimeHost(factory)
    with pytest.raises(RuntimeContractError, match="post-reset first observation"):
        host.open(spec())
    assert factory.reset_calls == 1
    assert factory.close_calls == 1


def test_identity_sources_cannot_be_collapsed():
    with pytest.raises(RuntimeContractError, match="sources must be distinct"):
        EnvironmentIdentity("e", 0, 1, "a" * 64, "b" * 64, "r", "same", "same")


def test_equal_digest_values_are_not_accepted_as_distinct_state_sources():
    with pytest.raises(RuntimeContractError, match="distinct from observation"):
        EnvironmentIdentity("e", 0, 1, "a" * 64, "a" * 64, "r")
