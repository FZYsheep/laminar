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
        config=GlobalRequestRouterConfig(
            logical_kv=LogicalKVConfig(
                enabled=True,
                growth_tokens_per_state=0,
                **policy,
            )
        ),
        kv_capacities={server: KVCapacity(capacity, 1, 10) for server in versions},
    )
    for server, version in versions.items():
        instance.mark_active(server, weight_version=version)
    return instance


@pytest.mark.parametrize("context_tokens", [0, 100, 490])
def test_session_limit_applies_independently_of_kv_demand(context_tokens):
    r = router({"a": 1}, max_sessions_per_replica=2)
    for i in range(2):
        r.bind_route(str(i), min_weight_version=1, context_tokens=context_tokens)
    assert isinstance(r.try_bind_route("waiting"), RouteUnavailable)
    assert r.get_active_session_count() == 2
    assert r.bind_route("0", min_weight_version=1) == 1
    r.release_route("0")
    assert r.try_bind_route("waiting", min_weight_version=1) == 1
    assert r.get_budget_snapshot()["a"]["session_count"] == 2


def test_startup_burst_is_bounded_per_replica():
    r = router({"a": 1, "b": 1, "c": 1})
    for i in range(72):
        assert r.try_bind_route(str(i), min_weight_version=1) == 1
    assert all(row["session_count"] == 24 for row in r.get_budget_snapshot().values())
    for i in range(72, 100):
        assert isinstance(r.try_bind_route(str(i)), RouteUnavailable)
    assert r.get_active_session_count() == 72


def test_admission_balances_session_demand_across_accepted_versions():
    r = router({"old": 1, "new": 2})
    assert r.bind_route("long", min_weight_version=1, context_tokens=800) == 2
    assert r.bind_route("short", min_weight_version=1, context_tokens=10) == 1
    assert r.bind_route("medium", min_weight_version=1, context_tokens=300) == 1
    assert r.get_budget_snapshot()["old"]["logical_kv_bytes"] == 330


def test_rejected_bind_does_not_reserve_capacity_and_oversized_demand_is_fatal():
    r = router({"a": 1}, admission_capacity_multiplier=1.0)
    r.bind_route("occupied", context_tokens=900)
    for _ in range(20):
        assert isinstance(r.try_bind_route("waiting", context_tokens=100), RouteUnavailable)
    assert r.get_active_session_count() == 1
    with pytest.raises(ValueError, match="exceeds"):
        r.try_bind_route("oversized", context_tokens=1200)
    r.release_route("occupied")
    assert r.try_bind_route("waiting", min_weight_version=1, context_tokens=100) == 1


def test_all_replicas_over_budget_can_finish_but_cannot_admit_or_merge():
    r = router({"a": 1, "b": 1, "c": 1})
    for i in range(72):
        r.bind_route(str(i), context_tokens=10)
    growing = []
    for server in ("a", "b", "c"):
        session = next(s for s in r._sessions.values() if s.route.server_id == server)
        growing.append(session.session_id)
        r.update_session_budget(session.session_id, context_tokens=560, state_count=1, revision=2)
        r.update_session_budget(session.session_id, context_tokens=1, state_count=1, revision=1)
    assert all(row["logical_kv_bytes"] == 1030 for row in r.get_budget_snapshot().values())
    assert isinstance(r.try_bind_route("new"), RouteUnavailable)
    assert not r.plan_and_begin_repack(loadable_version=2).groups
    for session_id in growing:
        assert r.acquire_lease(session_id, lease_id="finish")[0] == r._sessions[session_id].route.server_id
        r.release_lease(session_id, "finish")
        r.release_route(session_id)
    assert r.try_bind_route("new", min_weight_version=1) == 1


def test_owner_totals_include_idle_chains_and_allow_chain_removal():
    r = router({"a": 1})
    r.bind_route("session")
    r.update_session_budget("session", context_tokens=400, state_count=3, revision=1, reserved_tokens=60)
    assert r.get_budget_snapshot()["a"]["logical_kv_bytes"] == 490
    assert r.get_budget_snapshot()["a"]["inflight_requests"] == 0
    r.update_session_budget("session", context_tokens=100, state_count=1, revision=2)
    assert r.get_budget_snapshot()["a"]["logical_kv_bytes"] == 110


def test_repack_accumulates_demand_and_retains_original_request_leases():
    r = router({"a": 1, "b": 1, "c": 1, "d": 1}, capacity=100)
    for i in range(4):
        r.bind_route(str(i), min_weight_version=1, context_tokens=20)
    source = r.acquire_lease("0", lease_id="inflight", weight_version=1)[0]
    plan = r.plan_and_begin_repack(loadable_version=2)
    assert len(plan.groups) == 1 and len(plan.groups[0].sources) == 2
    assert r.get_budget_snapshot()[plan.groups[0].destination]["logical_kv_bytes"] == 90
    assert r.acquire_lease("0", lease_id="inflight", weight_version=1)[0] == source


def test_repack_has_separate_cumulative_session_and_backlog_limits():
    r = router({"a": 1, "b": 1, "c": 1, "d": 1}, max_sessions_per_replica=2)
    for i in range(4):
        r.bind_route(str(i))
    plan = r.plan_and_begin_repack(loadable_version=2)
    destination = plan.groups[0].destination
    assert r.get_budget_snapshot()[destination]["session_count"] == 3
    assert r._kv_policy.fits(0, 3, 1000, repack=True)
    assert not r._kv_policy.fits(0, 3, 1000)

    r = router({"a": 1, "b": 1, "c": 1}, repack_max_waiting_requests=2)
    for i in range(3):
        r.bind_route(str(i))
        r.acquire_lease(str(i), lease_id="busy")
    assert len(r.plan_and_begin_repack(loadable_version=2).groups[0].sources) == 1


def test_repack_recomputes_source_demand_in_destination_layout():
    r = router({"a": 1, "b": 1}, capacity=100)
    r._kv_capacities["b"] = KVCapacity(100, 10, 10)
    r.bind_route("small", context_tokens=5)
    r.bind_route("other", context_tokens=5)
    # Force distinct placements to exercise heterogeneous target layouts.
    from laminar.global_request_router.router import _RouteBinding

    r._sessions["small"].route = _RouteBinding("a", 1)
    r._sessions["other"].route = _RouteBinding("b", 1)
    plan = r.plan_and_begin_repack(loadable_version=2)
    assert plan.groups[0].destination == "a"
    assert r.get_budget_snapshot()["a"]["logical_kv_bytes"] == 30


def test_latest_version_continuations_can_rebind_after_growth():
    r = router({"a": 1, "b": 2, "c": 2})
    from laminar.global_request_router.router import _RouteBinding

    r._latest_version = True
    r.bind_route("moving", context_tokens=200)
    r._sessions["moving"].route = _RouteBinding("a", 1)
    r.bind_route("occupied", context_tokens=500)
    r._sessions["occupied"].route = _RouteBinding("b", 2)
    r.update_session_budget("moving", context_tokens=1400, state_count=1, revision=1)
    assert r.acquire_lease("moving", lease_id="next")[0] == "c"
    assert isinstance(r.try_bind_route("new", context_tokens=800), RouteUnavailable)


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


def test_worker_capacity_deduplicates_storage_and_accounts_for_rank_imbalance():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    tensor = torch.empty(100)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_caches=[tensor[:10], {"alias": tensor[20:]}, tensor],
            get_kv_cache_spec=lambda: {"attention": spec},
        ),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=2)),
    )
    capacity = worker_kv_capacity(worker)
    assert capacity.capacity_bytes == tensor.untyped_storage().nbytes()
    assert capacity.bytes_per_token == spec.page_size_bytes / 32
    workers = [
        {"capacity_bytes": 1000, "bytes_per_token": 10, "bytes_per_state": 5},
        {"capacity_bytes": 100, "bytes_per_token": 2, "bytes_per_state": 1},
    ]
    assert KVCapacity.from_workers(workers) == KVCapacity(100, 2, 1)
    with pytest.raises(ValueError, match="every worker"):
        KVCapacity.from_workers([])


def test_logical_policy_is_opt_in_and_requires_complete_capacity():
    r = GlobalRequestRouter(servers={"a": object()})
    r.mark_active("a", weight_version=1)
    for i in range(50):
        r.bind_route(str(i))
    assert r.get_budget_snapshot() == {}
    with pytest.raises(ValueError, match="every replica"):
        GlobalRequestRouter(servers={"a": object()}, config=GlobalRequestRouterConfig(logical_kv={"enabled": True}))


def test_training_config_retains_default_policy_and_can_enable_logical_admission():
    from pathlib import Path

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(__file__).parents[1] / "src/laminar/train/config/async_rl_base.yaml")
    assert not instantiate(config.rollout_router).logical_kv.enabled
    assert not config.rollout_dispatcher.logical_kv_admission
    config.rollout_router.logical_kv.enabled = True
    assert instantiate(config.rollout_router).logical_kv.max_sessions_per_replica == 24
    assert config.rollout_dispatcher.logical_kv_admission
