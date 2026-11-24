# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from laminar.global_request_router.kv_budget import (
    KVCapacity,
    LogicalKVConfig,
    capacity_from_cache_specs,
    worker_kv_capacity,
)
from laminar.global_request_router.router import GlobalRequestRouter, GlobalRequestRouterConfig, RouteUnavailable


def router(versions, *, capacity=1000, **policy):
    instance = GlobalRequestRouter(
        servers={server: object() for server in versions},
        config=GlobalRequestRouterConfig(logical_kv=LogicalKVConfig(enabled=True, growth_tokens_per_state=0, **policy)),
        kv_capacities={server: KVCapacity(capacity, 1, 10) for server in versions},
    )
    for server, version in versions.items():
        instance.mark_active(server, weight_version=version)
    return instance


def test_session_cap_applies_per_replica_and_releases_on_close():
    r = router({"a": 1, "b": 1}, max_sessions_per_replica=2)
    for i in range(4):
        r.bind_route(str(i))
    r.bind_route("0")
    assert all(row["session_count"] == 2 for row in r.get_budget_snapshot().values())
    assert isinstance(r.try_bind_route("waiting"), RouteUnavailable)
    assert r.get_active_session_count() == 4
    r.release_route("0")
    assert r.try_bind_route("waiting") is None
    assert r.get_active_session_count() == 4


def test_placement_balances_total_session_demand_across_accepted_versions():
    r = router({"old": 1, "new": 2})
    assert r.bind_route("long", min_weight_version=1, context_tokens=800) == 2
    assert r.bind_route("short", min_weight_version=1, context_tokens=10) == 1
    assert r.bind_route("medium", min_weight_version=1, context_tokens=300) == 1
    assert r.get_budget_snapshot()["old"]["logical_kv_bytes"] == 330


def test_growth_blocks_admission_and_repack_but_allows_existing_sessions_to_finish():
    r = router({"a": 1, "b": 1})
    for session_id in ("0", "1"):
        r.bind_route(session_id, context_tokens=10)
        r.update_session_budget(session_id, context_tokens=1250, state_count=1, revision=1)
    assert all(row["logical_kv_bytes"] == 1260 for row in r.get_budget_snapshot().values())
    assert isinstance(r.try_bind_route("new"), RouteUnavailable)
    assert not r.plan_and_begin_repack(loadable_version=2).groups
    for session_id in ("0", "1"):
        r.acquire_lease(session_id, lease_id="finish")
        r.release_lease(session_id, "finish")
        r.release_route(session_id)
    assert r.try_bind_route("new") is None


def test_repack_accumulates_kv_demand_and_retains_original_request_leases():
    r = router({"a": 1, "b": 1, "c": 1, "d": 1}, capacity=100)
    for i in range(4):
        r.bind_route(str(i), context_tokens=20)
    source = r.acquire_lease("0", lease_id="inflight")[0]
    plan = r.plan_and_begin_repack(loadable_version=2)
    assert len(plan.groups) == 1 and len(plan.groups[0].sources) == 2
    assert r.get_budget_snapshot()[plan.groups[0].destination]["logical_kv_bytes"] == 90
    assert r.acquire_lease("0", lease_id="inflight")[0] == source


@pytest.mark.parametrize(
    "policy,replicas,expected_sources",
    [({"max_sessions_per_replica": 2}, 4, 2), ({"repack_max_waiting_requests": 2}, 3, 1)],
)
def test_repack_enforces_cumulative_session_and_request_limits(policy, replicas, expected_sources):
    r = router({str(i): 1 for i in range(replicas)}, **policy)
    for i in range(replicas):
        r.bind_route(str(i))
        r.acquire_lease(str(i), lease_id="busy")
    plan = r.plan_and_begin_repack(loadable_version=2)
    assert len(plan.groups[0].sources) == expected_sources
    assert r.get_budget_snapshot()[plan.groups[0].destination]["session_count"] == expected_sources + 1


@pytest.mark.parametrize("mode", ["none", "align", "all"])
def test_hybrid_coefficients_follow_native_mamba_cache_mode(mode):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
        model_config=SimpleNamespace(max_model_len=1024),
    )
    attention = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    mamba = MambaSpec(block_size=16, shapes=((2, 4),), dtypes=(torch.float32,), mamba_cache_mode=mode)
    capacity = capacity_from_cache_specs({"attention": attention, "mamba": mamba}, config, 1_000_000)
    assert capacity.bytes_per_token == attention.page_size_bytes / 16 + (2 if mode == "all" else 0)
    assert capacity.bytes_per_state == attention.page_size_bytes + (64 if mode == "align" else 32)


def test_worker_capacity_counts_shared_storage_once_and_respects_rank_imbalance():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    tensor = torch.empty(100)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_caches=[tensor, tensor[:10]],
            get_kv_cache_spec=lambda: {"attention": spec},
        ),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=2)),
    )
    capacity = worker_kv_capacity(worker)
    assert capacity.capacity_bytes == tensor.untyped_storage().nbytes()
    assert capacity.bytes_per_token == spec.page_size_bytes / 32
    assert KVCapacity.from_workers(
        [
            {"capacity_bytes": 1000, "bytes_per_token": 10, "bytes_per_state": 5},
            {"capacity_bytes": 100, "bytes_per_token": 2, "bytes_per_state": 1},
        ]
    ) == KVCapacity(100, 2, 1)
