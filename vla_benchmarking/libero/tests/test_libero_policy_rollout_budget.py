from __future__ import annotations

import numpy as np

from vla_benchmarking.libero.evaluation.contracts import build_task_seed_matrix
from vla_benchmarking.libero.evaluation.libero_policy_rollout import (
    CallableLiberoEnvironmentFactory,
    DEFAULT_EPISODE_STEP_BUDGET,
    ProductionLiberoEnvironmentFactory,
    run_libero_policy_eval,
)
from vla_benchmarking.libero.evaluation.plan import build_evaluation_plan
from vla_benchmarking.libero.evaluation.policy_adapter import CallablePolicyAdapter, PolicyMetadata
from vla_benchmarking.libero.evaluation.run_policy_eval import EpisodeSpec


def _case(*, horizon: int = 10, plan_budget: int | None = None):
    metadata = PolicyMetadata(
        policy_kind="smolvla_no_arrow",
        artifact_id="artifact",
        checkpoint_revision="rev",
        backend="lerobot",
        native_action_horizon=horizon,
    )
    adapter = CallablePolicyAdapter(
        metadata,
        act_fn=lambda _: np.zeros((horizon, 7), dtype=np.float32),
    )
    plan = build_evaluation_plan(
        policy_kind="smolvla_no_arrow",
        suite_mode="vanilla",
        task_ids=[0],
        episodes_per_task=1,
    )
    if plan_budget is not None:
        # The runner validates the plan's existing digest before accepting it;
        # this test models a sealed externally-authored plan by recomputing it.
        from vla_benchmarking.libero.evaluation.plan import canonical_sha256

        plan["episode_step_budget"] = int(plan_budget)
        payload = dict(plan)
        payload.pop("sha256")
        plan["sha256"] = canonical_sha256(payload)
    cell = build_task_seed_matrix(task_ids=[0], episodes_per_task=1)[0]
    return adapter, plan, [EpisodeSpec(cell, "pick up the bowl")]


class _NeverTerminalEnv:
    def __init__(self, *, success_at: int | None = None):
        self.actions: list[np.ndarray] = []
        self.success_at = success_at
        self.closed = False

    def reset(self, *, seed: int, task_id: int, episode_index: int):
        assert (seed, task_id, episode_index) == (1000, 0, 0)
        return {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)}

    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
            0.0,
            False,
            False,
            {"success": self.success_at == len(self.actions)},
        )

    def close(self):
        self.closed = True


def test_default_budget_stops_mid_chunk_and_records_truncation():
    adapter, plan, episodes = _case(horizon=10)
    env = _NeverTerminalEnv()
    records = run_libero_policy_eval(
        adapter,
        plan,
        episodes,
        env_factory=CallableLiberoEnvironmentFactory(lambda _: env),
        episode_step_budget=3,
    )
    record = records[0]
    assert len(env.actions) == 3
    assert record.executed_actions == 3
    assert record.action_chunks == 1
    assert record.success is False
    assert record.failure_category == "episode_step_budget"
    assert record.metadata["environment_steps"] == 3
    assert record.metadata["environment_step_budget"] == 3
    assert record.metadata["budget_exhausted"] is True
    assert record.metadata["truncated"] is True
    assert env.closed is True


def test_success_on_budget_boundary_wins_over_budget_truncation():
    adapter, plan, episodes = _case(horizon=10)
    env = _NeverTerminalEnv(success_at=3)
    records = run_libero_policy_eval(
        adapter,
        plan,
        episodes,
        env_factory=CallableLiberoEnvironmentFactory(lambda _: env),
        episode_step_budget=3,
    )
    record = records[0]
    assert record.success is True
    assert record.failure_category is None
    assert record.executed_actions == 3
    assert record.metadata["budget_exhausted"] is False
    assert record.metadata["termination_reason"] == "success"


def test_plan_budget_overrides_shared_default_and_rejects_conflicting_argument():
    adapter, plan, episodes = _case(horizon=4, plan_budget=5)
    env = _NeverTerminalEnv()
    records = run_libero_policy_eval(
        adapter,
        plan,
        episodes,
        env_factory=CallableLiberoEnvironmentFactory(lambda _: env),
    )
    assert records[0].metadata["environment_step_budget"] == 5
    assert records[0].executed_actions == 5

    adapter, plan, episodes = _case(horizon=4, plan_budget=5)
    try:
        run_libero_policy_eval(
            adapter,
            plan,
            episodes,
            env_factory=CallableLiberoEnvironmentFactory(lambda _: _NeverTerminalEnv()),
            episode_step_budget=3,
        )
    except ValueError as exc:
        assert "sealed evaluation plan" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("conflicting sealed budget was accepted")


def test_shared_default_is_sealed_at_220():
    assert DEFAULT_EPISODE_STEP_BUDGET == 220


def test_default_budget_is_not_shortened_by_four_step_native_horizon():
    adapter, plan, episodes = _case(horizon=4)
    env = _NeverTerminalEnv()
    records = run_libero_policy_eval(
        adapter,
        plan,
        episodes,
        env_factory=CallableLiberoEnvironmentFactory(lambda _: env),
    )
    assert records[0].executed_actions == DEFAULT_EPISODE_STEP_BUDGET
    assert records[0].metadata["budget_exhausted"] is True


def test_production_factory_preserves_post_setup_state_without_second_reset(monkeypatch):
    import vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval as production

    adapter, plan, episodes = _case(horizon=4)
    del adapter, plan
    episode = episodes[0]

    class FakeProductionEnv:
        def __init__(self):
            # build_libero_env has already performed the one setup reset.
            self.reset_calls = 1
            self.closed = False
            self._arrow_init_state_diagnostics = {
                "source": "test",
                "selected_index": int(episode.cell.init_state_index),
            }
            self._arrow_environment_audit = {
                "scene_randomization": "vanilla",
                "applied_removals": [],
                "applied_swaps": [],
            }

        def reset(self):
            self.reset_calls += 1
            return {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)}

        def _get_observations(self, force_update=True):
            assert force_update is True
            return {"agentview": np.ones((256, 256, 3), dtype=np.uint8)}

        def close(self):
            self.closed = True

    fake = FakeProductionEnv()
    captured = {}

    def build_libero_env(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake

    monkeypatch.setattr(production, "build_libero_env", build_libero_env, raising=False)
    factory = ProductionLiberoEnvironmentFactory()
    environment = factory(episode)
    assert fake.reset_calls == 1
    observation = environment.reset(
        seed=episode.cell.seed,
        task_id=episode.cell.task_id,
        episode_index=episode.cell.episode_index,
    )
    assert fake.reset_calls == 1
    assert int(observation["agentview"][0, 0, 0]) == 1
    assert environment.setup_audit["init_state"]["selected_index"] == 0
    assert captured["kwargs"]["init_state_index"] == 0
    environment.close()
    assert fake.closed is True
