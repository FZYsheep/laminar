# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.rollout.replica import TokenOutput

from laminar.global_request_router import GlobalRequestRouter
from laminar.global_request_router.router import ACTIVE, DRAINING
from laminar.llm_server_client import LLMServerClient
from laminar.multi_version_rollout import (
    BestFitRepackPolicy,
    CommittedRepackGroup,
    MultiVersionRolloutRepackConfig,
    RepackPolicy,
    RepackReplica,
    RepackSession,
)


def _snapshot(server_id: str, kv: float | None, *, version: int = 1) -> dict[str, object]:
    return {
        "server_id": server_id,
        "backend_health": "healthy",
        "kv_cache_usage": kv,
        "num_waiting_requests": 0,
        "num_running_requests": 0,
        "snapshot_version": version,
    }


class _RecordingPolicy:
    def __init__(self, *, warmup_passes: int = 0) -> None:
        self.warmup_passes = warmup_passes
        self.seen: list[tuple[RepackReplica, ...]] = []

    def plan(self, replicas: Sequence[RepackReplica]) -> Mapping[str, Sequence[str]]:
        self.seen.append(tuple(replicas))
        by_id = {r.server_id: r for r in replicas}
        if len(self.seen) > self.warmup_passes and {"s0", "s1"} <= by_id.keys() and by_id["s0"].sessions:
            return {"s1": ("s0",)}
        return {}


@pytest.mark.asyncio
async def test_custom_policy_resolution_and_session_telemetry() -> None:
    assert isinstance(BestFitRepackPolicy(), RepackPolicy)
    with pytest.raises(ValueError, match="missing or invalid kv_cache_usage"):
        BestFitRepackPolicy().plan([RepackReplica("s0", 1, None), RepackReplica("s1", 1, 0.1)])
    bfd = BestFitRepackPolicy(source_max_kv_cache_usage=0.45, destination_max_kv_cache_usage=0.9)
    assert bfd.plan(
        [
            RepackReplica(sid, 1, kv, sessions=(RepackSession(sid),))
            for sid, kv in (("s0", 0.2), ("s1", 0.4), ("s2", 0.5))
        ]
    ) == {"s2": ("s1",)}

    repack_cfg: MultiVersionRolloutRepackConfig = omega_conf_to_dataclass(
        OmegaConf.create(
            {
                "_target_": "laminar.multi_version_rollout.MultiVersionRolloutRepackConfig",
                "policy": {"_target_": f"{__name__}._RecordingPolicy", "warmup_passes": 1},
            }
        )
    )
    assert isinstance(repack_cfg.policy, _RecordingPolicy)

    async def _gen(**_kwargs):
        return TokenOutput(token_ids=[10, 11, 12, 13], stop_reason="completed", extra_fields={"global_steps": 1})

    servers = {sid: SimpleNamespace(generate=SimpleNamespace(remote=_gen)) for sid in ("s0", "s1")}
    router = GlobalRequestRouter(servers=servers, repack_policy=repack_cfg.policy)
    for sid in servers:
        router.mark_active(sid, weight_version=1)

    def _remote(method_name: str):
        async def _call(*args, **kwargs):
            return getattr(router, method_name)(*args, **kwargs)

        return SimpleNamespace(remote=_call)

    client = LLMServerClient(
        config=SimpleNamespace(),
        router=SimpleNamespace(
            **{m: _remote(m) for m in ("bind_route", "acquire_lease", "release_lease", "release_route")}
        ),
    )
    for idx, sid in enumerate(("s0", "s1"), start=1):
        router.update_routing_snapshot({s: _snapshot(s, 0.0 if s == sid else 0.9, version=idx) for s in servers})
        await client.bind_route(f"sess-{sid}", weight_version=1)
        await client.generate(
            session_id=f"sess-{sid}", weight_version=1, prompt_ids=list(range(56)), sampling_params={}
        )

    # Custom policy governs even above default 0.5/0.95 KV limits and retains state across passes.
    router.update_routing_snapshot({s: _snapshot(s, 0.85, version=10) for s in servers})
    assert router.plan_and_begin_repack(loadable_version=2).groups == ()
    by_id = {r.server_id: r for r in repack_cfg.policy.seen[0]}
    assert by_id["s0"].sessions[0].context_tokens == 60

    plan = router.plan_and_begin_repack(loadable_version=2)
    assert plan.groups == (CommittedRepackGroup(sources=("s0",), destination="s1", weight_version=1),)
    assert router._server_states["s0"].lifecycle == DRAINING
    assert router._session_id_to_server["sess-s0"].server_id == "s1"
    assert len(repack_cfg.policy.seen) == 2


@pytest.mark.parametrize(
    ("proposal", "match"),
    [
        ({"s2": ()}, "empty sources"),
        ({"s2": ("s0", "s0")}, "duplicate sources"),
        ({"s0": ("s0",)}, "both source and destination"),
        ({"s2": ("unknown",)}, "not in the eligible version cohort"),
        ({"s2": ("idle",)}, "no bound sessions"),
        ({"s2": ("s0",), "s1": ("s2",)}, "overlap"),
    ],
)
def test_router_validates_policy_proposals_before_mutating_routes(
    proposal: dict[str, tuple[str, ...]], match: str
) -> None:
    router = GlobalRequestRouter(
        servers={sid: object() for sid in ("s0", "s1", "s2", "idle")},
        repack_policy=SimpleNamespace(plan=lambda replicas: proposal),
    )
    for step, sid in enumerate(("s0", "s1", "s2", "idle"), start=1):
        router.mark_active(sid, weight_version=1)
        if sid != "idle":
            router.update_routing_snapshot(
                {s: _snapshot(s, 0.0 if s == sid else 0.9, version=step) for s in router._server_states}
            )
            router.bind_route(f"{sid}-sess", min_weight_version=1)

    before_routes = dict(router._session_id_to_server)
    with pytest.raises(ValueError, match=match):
        router.plan_and_begin_repack(loadable_version=2)
    assert router._session_id_to_server == before_routes
    assert all(state.lifecycle == ACTIVE for state in router._server_states.values())
