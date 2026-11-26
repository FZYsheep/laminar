# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import ray
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.rollout.replica import TokenOutput

from laminar.eval.router import ValidationRequestRouter
from laminar.global_request_router import (
    GlobalRequestRouter,
    GlobalRequestRouterConfig,
    ReplicaConcurrencyConfig,
    RouteUnavailable,
    RouteUnavailableError,
)
from laminar.llm_server_client import LLMServerClient


def _snapshot(server_id: str, *, kv: float = 0.05, version: int = 1, healthy: bool = True) -> dict[str, Any]:
    return {
        "server_id": server_id,
        "backend_health": "healthy" if healthy else "unavailable",
        "kv_cache_usage": kv,
        "num_waiting_requests": 0,
        "num_running_requests": 0,
        "snapshot_version": version,
        "timestamp": float(version),
    }


def _router(
    versions: dict[str, int],
    *,
    continuation: str = "same_version",
    replica_concurrency: ReplicaConcurrencyConfig | dict[str, Any] | None = None,
    kv_by_server: dict[str, float] | None = None,
) -> GlobalRequestRouter:
    router = GlobalRequestRouter(
        servers={server_id: SimpleNamespace(server_id=server_id) for server_id in versions},
        config=GlobalRequestRouterConfig(
            continuation=continuation,
            replica_concurrency=ReplicaConcurrencyConfig() if replica_concurrency is None else replica_concurrency,
        ),
    )
    for server_id, version in versions.items():
        router.mark_active(server_id, weight_version=version)
    router.update_routing_snapshot(
        {server_id: _snapshot(server_id, kv=(kv_by_server or {}).get(server_id, 0.05)) for server_id in versions}
    )
    return router


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        result = self._fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result


@pytest.fixture(scope="module")
def local_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")
    yield
    ray.shutdown()


@pytest.mark.parametrize(
    "config_module,config_name", [("laminar.train.config", "async_rl"), ("laminar.eval.config", "validation")]
)
def test_hydra_base_configs_expose_replica_concurrency_defaults_and_overrides(
    config_module: str,
    config_name: str,
) -> None:
    with initialize_config_module(config_module=config_module, version_base=None):
        default_cfg = compose(config_name=config_name)
        overridden_cfg = compose(
            config_name=config_name,
            overrides=[
                "rollout_router.replica_concurrency.max_sessions=14",
                "rollout_router.replica_concurrency.repack_max_sessions=20",
                "rollout_router.replica_concurrency.max_inflight_requests=8",
            ],
        )
    parsed_default = omega_conf_to_dataclass(default_cfg.rollout_router)
    assert parsed_default.replica_concurrency == ReplicaConcurrencyConfig()
    assert not parsed_default.replica_concurrency.enabled

    parsed_override = omega_conf_to_dataclass(overridden_cfg.rollout_router)
    assert parsed_override.replica_concurrency == ReplicaConcurrencyConfig(
        max_sessions=14,
        repack_max_sessions=20,
        max_inflight_requests=8,
    )
    assert parsed_override.replica_concurrency.enabled
    assert parsed_override.replica_concurrency.effective_repack_max_sessions == 20


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"max_sessions": 0},
        {"max_sessions": -2},
        {"max_sessions": True},
        {"repack_max_sessions": 0},
        {"max_inflight_requests": 1.5},
    ],
)
def test_replica_concurrency_config_rejects_non_positive_or_non_int_values(bad_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="replica_concurrency"):
        GlobalRequestRouterConfig(replica_concurrency=bad_kwargs)


def test_exact_version_max_sessions_gates_admission_and_advances_to_newer_version() -> None:
    # Two replicas: s0 at v1, s1 at v2; each capped at 2 bound sessions.
    router = _router({"s0": 1, "s1": 2}, replica_concurrency={"max_sessions": 2})

    assert router.get_routable_version(1) == 1
    assert router.bind_route("a", min_weight_version=1) == 1
    assert router.bind_route("b", min_weight_version=1) == 1
    # Idempotent rebind of an already admitted session never consumes a third slot.
    assert router.bind_route("a", min_weight_version=1) == 1
    assert router.get_replica_concurrency()["s0"]["bound_sessions"] == 2

    # Once v1 replica s0 is full, minimum=1 automatically routes to v2 replica s1.
    assert router.get_routable_version(1) == 2
    assert router.bind_route("c", min_weight_version=1) == 2
    assert router.bind_route("d", min_weight_version=1) == 2

    # All replicas >= 1 are now at capacity: both version check and new bind wait.
    with pytest.raises(RouteUnavailableError):
        router.get_routable_version(1)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("overflow", min_weight_version=1)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("unversioned-overflow")

    # Admitted sessions continue acquiring leases normally while the pool is full.
    served_by, _ = router.acquire_lease("a", lease_id="turn-1", weight_version=1)
    assert served_by == "s0"
    router.release_lease("a", "turn-1")
    router.release_route("a", weight_version=1)

    # Releasing one slot on s0 reopens v1 admission immediately.
    assert router.get_routable_version(1) == 1
    assert router.bind_route("overflow", min_weight_version=1) == 1


def test_latest_version_preserves_fenced_session_quota_across_colocate_pause() -> None:
    router = _router(
        {"s0": 1, "s1": 1},
        continuation="latest_version",
        replica_concurrency={"max_sessions": 1},
    )
    router.bind_route("existing-0", min_weight_version=1)
    router.bind_route("existing-1", min_weight_version=1)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("newcomer", min_weight_version=1)

    # Colocate pause drops routes while keeping live sessions in the Router.
    router.begin_colocate_pause()
    router.mark_active("s0", weight_version=2)
    router.mark_active("s1", weight_version=2)
    assert router._session_id_to_server == {}
    assert router.get_active_session_count() == 2

    # Newcomers cannot steal the two slots reserved by unbound live sessions.
    with pytest.raises(RouteUnavailableError):
        router.get_routable_version(2)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("newcomer", min_weight_version=2)

    # Fenced live sessions rebind without deadlock on their next turn and finish.
    s_a, _ = router.acquire_lease("existing-0", lease_id="resume-0")
    s_b, _ = router.acquire_lease("existing-1", lease_id="resume-1")
    assert {s_a, s_b} == {"s0", "s1"}
    router.release_lease("existing-0", "resume-0")
    router.release_route("existing-0")

    assert router.get_routable_version(2) == 2
    assert router.bind_route("newcomer", min_weight_version=2) is None


def test_repack_honors_default_and_explicit_destination_session_caps() -> None:
    # Default repack_max_sessions inherits max_sessions=3: s0(1)+s1(1)+s2(2)=4 > 3,
    # so only one source can merge onto s2 even though KV usage is tiny (0.05 each).
    router = _router(
        {"s0": 1, "s1": 1, "s2": 1},
        replica_concurrency={"max_sessions": 3},
        kv_by_server={"s0": 0.05, "s1": 0.05, "s2": 0.05},
    )
    router.bind_route("s0-a", min_weight_version=1)
    router.bind_route("s1-a", min_weight_version=1)
    router.bind_route("s2-a", min_weight_version=1)
    router.bind_route("s2-b", min_weight_version=1)
    # Place explicitly as 1, 1, 2 across s0, s1, s2.
    router._sessions["s0-a"].route = router._sessions["s0-a"].route.__class__("s0", 1)
    router._sessions["s1-a"].route = router._sessions["s1-a"].route.__class__("s1", 1)
    router._sessions["s2-a"].route = router._sessions["s2-a"].route.__class__("s2", 1)
    router._sessions["s2-b"].route = router._sessions["s2-b"].route.__class__("s2", 1)

    plan = router.plan_and_begin_repack(loadable_version=2)
    assert all(info["bound_sessions"] <= 3 for info in router.get_replica_concurrency().values())
    # 4 total sessions across 3 replicas with cap=3 requires at least 2 destinations, so 1 source drains.
    assert sum(len(group.sources) for group in plan.groups) == 1

    # Explicit repack_max_sessions=4 allows consolidating both s0 and s1 onto s2 in one group.
    relaxed = _router(
        {"s0": 1, "s1": 1, "s2": 1},
        replica_concurrency={"max_sessions": 2, "repack_max_sessions": 4},
        kv_by_server={"s0": 0.05, "s1": 0.05, "s2": 0.10},
    )
    for sid, server_id in (("a", "s0"), ("b", "s1"), ("c", "s2"), ("d", "s2")):
        relaxed.bind_route(sid, min_weight_version=1)
        relaxed._sessions[sid].route = relaxed._sessions[sid].route.__class__(server_id, 1)
    relaxed_plan = relaxed.plan_and_begin_repack(loadable_version=2)
    assert len(relaxed_plan.groups) == 1
    assert relaxed_plan.groups[0].destination == "s2"
    assert relaxed_plan.groups[0].sources == ("s0", "s1")
    assert relaxed.get_replica_concurrency()["s2"]["bound_sessions"] == 4


def test_validation_router_enforces_per_replica_max_sessions_within_group() -> None:
    router = ValidationRequestRouter(
        servers={"s0": SimpleNamespace(), "s1": SimpleNamespace()},
        config=GlobalRequestRouterConfig(replica_concurrency={"max_sessions": 1}),
    )
    for server_id, version in (("s0", 1), ("s1", 2)):
        router.mark_active(server_id, weight_version=version)
    router.update_routing_snapshot({"s0": _snapshot("s0"), "s1": _snapshot("s1")})
    router.set_validation_versions([1, 2])

    assert router.bind_route("val-v1-0", min_weight_version=1) == 1
    with pytest.raises(RouteUnavailableError):
        router.bind_route("val-v1-1", min_weight_version=1)
    # Invalid version still raises ValueError rather than waiting.
    with pytest.raises(ValueError, match="not assigned"):
        router.bind_route("invalid", min_weight_version=99)
    # Group v2 still has independent per-replica capacity.
    assert router.bind_route("val-v2-0", min_weight_version=2) == 2


@pytest.mark.asyncio
async def test_total_semaphore_and_per_replica_caps_cooperate_without_uni_agent_changes() -> None:
    """Simulate uni_agent's global runner semaphore + LLMServerClient bind/generate/release."""
    peak_sessions = {"s0": 0, "s1": 0}
    peak_inflight = {"s0": 0, "s1": 0}
    gate = asyncio.Event()

    class _Server:
        def __init__(self, server_id: str) -> None:
            self.server_id = server_id
            self.generate = _Remote(self._generate)

        async def _generate(self, **_kwargs: Any) -> TokenOutput:
            snap = router.get_replica_concurrency()[self.server_id]
            peak_sessions[self.server_id] = max(peak_sessions[self.server_id], snap["bound_sessions"])
            peak_inflight[self.server_id] = max(peak_inflight[self.server_id], snap["inflight_requests"])
            await gate.wait()
            await asyncio.sleep(0.005)
            return TokenOutput(
                token_ids=[1, 2], log_probs=[-0.1, -0.1], stop_reason="completed", extra_fields={"global_steps": 1}
            )

    servers = {"s0": _Server("s0"), "s1": _Server("s1")}
    router_cfg = GlobalRequestRouterConfig(
        client_wait={
            "timeout_s": 5.0,
            "planned_pause_timeout_s": 10.0,
            "initial_backoff_s": 0.005,
            "max_backoff_s": 0.02,
        },
        replica_concurrency={"max_sessions": 2, "max_inflight_requests": 1},
    )
    router = GlobalRequestRouter(servers=servers, config=router_cfg)
    # Skewed rollout state: only s0 is active at v1 initially; s1 joins later.
    router.mark_active("s0", weight_version=1)
    router.update_routing_snapshot({"s0": _snapshot("s0"), "s1": _snapshot("s1")})

    router_proxy = SimpleNamespace(
        **{
            name: _Remote(getattr(router, name))
            for name in (
                "bind_route",
                "try_bind_route",
                "acquire_lease",
                "try_acquire_lease",
                "release_lease",
                "release_route",
            )
        }
    )
    client = LLMServerClient(
        config=OmegaConf.create(
            {
                "rollout_router": {
                    "client_wait": {
                        "timeout_s": 5.0,
                        "planned_pause_timeout_s": 10.0,
                        "initial_backoff_s": 0.005,
                        "max_backoff_s": 0.02,
                    },
                    "replica_concurrency": {"max_sessions": 2, "max_inflight_requests": 1},
                }
            }
        ),
        router=router_proxy,
    )

    # Global runner semaphore allows 6 concurrent episodes, exceeding single-replica capacity (2).
    total_semaphore = asyncio.Semaphore(6)

    async def run_episode(idx: int) -> None:
        session_id = f"ep-{idx}"
        async with total_semaphore:
            bound = await client.bind_route(session_id=session_id, weight_version=1)
            try:
                snap = router.get_replica_concurrency()
                for sid in ("s0", "s1"):
                    peak_sessions[sid] = max(peak_sessions[sid], snap[sid]["bound_sessions"])
                await client.generate(
                    session_id=session_id,
                    prompt_ids=[10, 11],
                    sampling_params={"max_new_tokens": 4},
                    weight_version=bound,
                )
            finally:
                await router_proxy.release_route.remote(session_id=session_id, weight_version=bound)

    episodes = [asyncio.create_task(run_episode(i)) for i in range(6)]
    await asyncio.sleep(0.03)
    # While gate is closed and only s0 is active, s0 holds at most 2 sessions and 1 in-flight request;
    # the remaining 4 episodes wait cleanly in bind_route despite holding global semaphore slots.
    assert router.get_replica_concurrency()["s0"]["bound_sessions"] == 2
    assert router.get_replica_concurrency()["s0"]["inflight_requests"] == 1
    assert router.get_replica_concurrency()["s1"]["bound_sessions"] == 0

    # Bring s1 online and open generation gate; all 6 episodes complete without violating caps.
    router.mark_active("s1", weight_version=1)
    gate.set()
    await asyncio.gather(*episodes)

    assert peak_sessions["s0"] <= 2 and peak_sessions["s1"] <= 2
    assert peak_inflight["s0"] <= 1 and peak_inflight["s1"] <= 1
    assert router.get_active_session_count() == 0


def test_ray_try_bind_and_try_acquire_wait_without_blocking_router_control(local_ray) -> None:
    router = ray.remote(GlobalRequestRouter).remote(
        servers={"s0": None},
        config=GlobalRequestRouterConfig(
            replica_concurrency={"max_sessions": 1, "max_inflight_requests": 1},
        ),
    )
    try:
        ray.get(router.mark_active.remote("s0", weight_version=1), timeout=5)
        assert ray.get(router.try_bind_route.remote("s-0", min_weight_version=1), timeout=5) == 1
        waits = ray.get(
            [router.try_bind_route.remote(f"wait-{i}", min_weight_version=1) for i in range(64)], timeout=10
        )
        assert all(isinstance(item, RouteUnavailable) for item in waits)

        # In-flight lease cap returns RouteUnavailable on a second concurrent turn.
        ray.get(router.try_acquire_lease.remote("s-0", lease_id="l-0", weight_version=1), timeout=5)
        second_lease = ray.get(router.try_acquire_lease.remote("s-0", lease_id="l-1", weight_version=1), timeout=5)
        assert isinstance(second_lease, RouteUnavailable)
        # Idempotent retry of the committed lease succeeds immediately.
        assert ray.get(router.try_acquire_lease.remote("s-0", lease_id="l-0", weight_version=1), timeout=5)[0] == "s0"

        ray.get(router.release_lease.remote("s-0", "l-0"), timeout=5)
        ray.get(router.release_route.remote("s-0", weight_version=1), timeout=5)
        assert ray.get(router.try_bind_route.remote("wait-0", min_weight_version=1), timeout=5) == 1
    finally:
        ray.kill(router, no_restart=True)
