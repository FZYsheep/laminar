# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, RouteUnavailableError
from laminar.llm_server_client import LLMServerClient


def _router(
    versions: dict[str, int], *, continuation: str = "same_version", max_sessions: int = 2
) -> GlobalRequestRouter:
    router = GlobalRequestRouter(
        servers={sid: SimpleNamespace(server_id=sid) for sid in versions},
        config=GlobalRequestRouterConfig(continuation=continuation, replica_concurrency={"max_sessions": max_sessions}),
    )
    for sid, version in versions.items():
        router.mark_active(sid, weight_version=version)
    router.update_routing_snapshot(
        {sid: {"backend_health": "healthy", "kv_cache_usage": 0.05, "snapshot_version": 1} for sid in versions}
    )
    return router


def test_max_sessions_gates_admission_without_restricting_repack() -> None:
    with pytest.raises(ValueError, match="replica_concurrency"):
        GlobalRequestRouterConfig(replica_concurrency={"max_sessions": 0})
    with pytest.raises(ValueError, match="unknown replica_concurrency"):
        GlobalRequestRouterConfig(replica_concurrency={"extra": 1})

    router = _router({"s0": 1, "s1": 2}, max_sessions=2)
    assert [router.bind_route("a", min_weight_version=1), router.bind_route("b", min_weight_version=1)] == [1, 1]
    assert router.bind_route("a", min_weight_version=1) == 1  # idempotent rebind
    assert router.get_routable_version(1) == 2
    assert [router.bind_route("c", min_weight_version=1), router.bind_route("d", min_weight_version=1)] == [2, 2]
    with pytest.raises(RouteUnavailableError):
        router.get_routable_version(1)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("overflow", min_weight_version=1)

    router.release_route("a", weight_version=1)
    assert router.bind_route("overflow", min_weight_version=1) == 1

    # Same-version Repack destination is intentionally unconstrained by admission max_sessions.
    repack_router = _router({"s0": 1, "s1": 1}, max_sessions=1)
    repack_router.bind_route("x0", min_weight_version=1)
    repack_router.bind_route("x1", min_weight_version=1)
    plan = repack_router.plan_and_begin_repack(loadable_version=2)
    assert len(plan.groups) == 1


@pytest.mark.asyncio
async def test_latest_fenced_quota_and_client_wait_backpressure() -> None:
    router = _router({"s0": 1, "s1": 1}, continuation="latest_version", max_sessions=1)
    router.bind_route("s-0", min_weight_version=1)
    router.bind_route("s-1", min_weight_version=1)
    router.begin_colocate_pause()
    router.mark_active("s0", weight_version=2)
    router.mark_active("s1", weight_version=2)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("newcomer", min_weight_version=2)
    assert {router.acquire_lease("s-0", lease_id="r0")[0], router.acquire_lease("s-1", lease_id="r1")[0]} == {
        "s0",
        "s1",
    }

    async def remote_bind(**kwargs):
        return router.bind_route(**kwargs)

    cfg = GlobalRequestRouterConfig(
        client_wait={"initial_backoff_s": 0.005, "max_backoff_s": 0.01}, replica_concurrency={"max_sessions": 1}
    )
    proxy = SimpleNamespace(bind_route=SimpleNamespace(remote=remote_bind))
    waiter = asyncio.create_task(LLMServerClient(SimpleNamespace(rollout_router=cfg), proxy).bind_route("newcomer"))
    await asyncio.sleep(0.02)
    assert not waiter.done()
    router.release_lease("s-0", "r0")
    router.release_route("s-0")
    assert await asyncio.wait_for(waiter, timeout=1.0) is None
