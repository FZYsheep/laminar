# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from verl.workers.rollout.replica import TokenOutput

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, KVCapacity, LogicalKVConfig
from laminar.llm_server.controller import LLMServerController
from laminar.llm_server_client import LLMServerClient
from laminar.train.dispatcher import RolloutDispatcher, RolloutDispatcherConfig


class Remote:
    def __init__(self, method):
        self.method = method

    async def remote(self, *args, **kwargs):
        return self.method(*args, **kwargs)


def client_with_router(*, outputs=None):
    calls = []
    outputs = list(
        outputs or [TokenOutput(token_ids=[7, 8], stop_reason="completed", extra_fields={"global_steps": 1})]
    )

    def generate(**kwargs):
        calls.append(kwargs)
        return outputs.pop(0)

    policy = LogicalKVConfig(enabled=True, max_sessions_per_replica=1, growth_tokens_per_state=3)
    config = GlobalRequestRouterConfig(
        logical_kv=policy,
        client_wait={"timeout_s": 1, "initial_backoff_s": 0.001, "max_backoff_s": 0.001},
    )
    router = GlobalRequestRouter(
        servers={"a": SimpleNamespace(generate=Remote(generate))},
        config=config,
        kv_capacities={"a": KVCapacity(1000, 1, 10)},
    )
    router.mark_active("a", weight_version=1)
    proxy = SimpleNamespace(
        **{
            name: Remote(getattr(router, name))
            for name in (
                "try_bind_route",
                "try_acquire_lease",
                "release_lease",
                "release_route",
                "update_session_budget",
            )
        }
    )
    client = LLMServerClient(config=SimpleNamespace(rollout_router=config), router=proxy)
    return client, router, calls


@pytest.mark.asyncio
async def test_capacity_wait_resumes_when_an_idle_session_closes():
    client, router, _ = client_with_router()
    await client.bind_route("occupied", weight_version=1)
    pending = asyncio.create_task(client.bind_route("next", weight_version=1, context_tokens=50))
    await asyncio.sleep(0.01)
    assert not pending.done()
    assert router.get_active_session_count() == 1
    router.release_route("occupied")
    assert await asyncio.wait_for(pending, timeout=1) == 1
    assert router.get_budget_snapshot()["a"]["logical_kv_bytes"] == 63


@pytest.mark.asyncio
async def test_aborted_generation_keeps_prompt_and_output_budget_until_close():
    outputs = [
        TokenOutput(token_ids=[7], stop_reason="aborted", extra_fields={"global_steps": 1}),
        TokenOutput(token_ids=[8], stop_reason="completed", extra_fields={"global_steps": 1}),
    ]
    client, router, calls = client_with_router(outputs=outputs)
    await client.bind_route("session", weight_version=1)
    output = await client.generate(
        session_id="session",
        weight_version=1,
        prompt_ids=[1, 2],
        sampling_params={"max_tokens": 4},
    )
    assert output.token_ids == [7, 8]
    assert [call["prompt_ids"] for call in calls] == [[1, 2], [1, 2, 7]]
    snapshot = router.get_budget_snapshot()["a"]
    assert snapshot["context_tokens"] == 4 and snapshot["reserved_tokens"] == 3
    assert snapshot["logical_kv_bytes"] == 17 and snapshot["inflight_requests"] == 0
    assert snapshot["session_count"] == 1
    await client.release_route("session")
    assert router.get_budget_snapshot()["a"]["logical_kv_bytes"] == 0


@pytest.mark.asyncio
async def test_owner_totals_survive_per_request_accounting():
    client, router, calls = client_with_router()
    await client.bind_route("chains", weight_version=1)
    await client.update_session_budget("chains", context_tokens=700, state_count=2, revision=2)
    await client.update_session_budget("chains", context_tokens=1, state_count=1, revision=1)
    await client.generate(
        session_id="chains", weight_version=1, prompt_ids=[1, 2], sampling_params={}, budget_managed=True
    )
    snapshot = router.get_budget_snapshot()["a"]
    assert snapshot["context_tokens"] == 700 and snapshot["state_count"] == 2
    assert snapshot["reserved_tokens"] == 6 and snapshot["logical_kv_bytes"] == 726
    assert "budget_managed" not in calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("logical_kv", [False, True])
async def test_controller_qualifies_capacity_before_colocated_sleep(logical_kv):
    events = []

    async def initialize(**kwargs):
        events.append("initialize")

    async def capacity_rpc(method):
        events.append(method)
        return [asdict(KVCapacity(1000, 10, 5)), asdict(KVCapacity(100, 2, 1))]

    async def sleep():
        events.append("sleep")

    engine = SimpleNamespace(
        init_colocated=initialize,
        _server_handle=SimpleNamespace(collective_rpc=SimpleNamespace(remote=capacity_rpc)),
    )
    controller = LLMServerController(relay_client=None, engine=engine, name="a")
    controller.sleep_for_training = sleep
    await controller.init_engine(resource_pool=None, colocate=True, logical_kv=logical_kv)
    assert events == (["initialize", "logical_kv_capacity", "sleep"] if logical_kv else ["initialize", "sleep"])
    if logical_kv:
        assert controller.get_kv_capacity() == KVCapacity(100, 2, 1)
    else:
        with pytest.raises(RuntimeError, match="not collected"):
            controller.get_kv_capacity()


@pytest.mark.asyncio
async def test_dispatcher_defers_admission_to_router_when_logical_policy_enabled():
    dispatcher = object.__new__(RolloutDispatcher)
    dispatcher.config = RolloutDispatcherConfig(
        schedule_interval_s=1,
        state_batch_size=1,
        num_samples_per_prompt=1,
        logical_kv_admission=True,
        pressure_admission_mode="enforce",
    )
    assert await dispatcher._llm_server_pressure_open()
