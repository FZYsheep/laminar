# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Logical session demand without discounts for shared or evictable prefix cache."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LogicalKVConfig:
    enabled: bool = False
    admission_capacity_multiplier: float = 1.2
    repack_capacity_multiplier: float = 1.0
    max_sessions_per_replica: int = 24
    repack_session_multiplier: float = 1.5
    growth_tokens_per_state: int = 512
    repack_max_waiting_requests: int = 32

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("logical_kv.enabled must be a boolean")
        for name in ("admission_capacity_multiplier", "repack_capacity_multiplier", "repack_session_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.repack_capacity_multiplier > self.admission_capacity_multiplier:
            raise ValueError("require repack capacity <= admission capacity")
        for name in ("max_sessions_per_replica", "growth_tokens_per_state", "repack_max_waiting_requests"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "max_sessions_per_replica" else 0):
                raise ValueError(f"{name} has an invalid integer value")

    @classmethod
    def from_value(cls, value):
        return value if isinstance(value, cls) else cls(**dict(value or {}))

    def fits(self, demand: float, sessions: int, capacity: float, *, repack: bool = False) -> bool:
        limit = self.repack_capacity_multiplier if repack else self.admission_capacity_multiplier
        session_limit = self.max_sessions_per_replica * (self.repack_session_multiplier if repack else 1)
        return sessions <= session_limit and demand <= limit * capacity


@dataclass(frozen=True)
class KVCapacity:
    capacity_bytes: float
    bytes_per_token: float
    bytes_per_state: float

    def __post_init__(self):
        for name in ("capacity_bytes", "bytes_per_token", "bytes_per_state"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.capacity_bytes == 0 or self.bytes_per_token + self.bytes_per_state == 0:
            raise ValueError("KV capacity and model coefficients must be positive")

    def demand(self, tokens: int, states: int, reserved_tokens: int) -> float:
        return self.bytes_per_token * (tokens + reserved_tokens) + self.bytes_per_state * states

    @classmethod
    def from_workers(cls, inventory: list[Mapping[str, Any]]) -> "KVCapacity":
        if not inventory:
            raise ValueError("logical KV admission requires capacity from every worker")
        workers = [cls(**worker) for worker in inventory]
        # Normalize to the most constrained rank, including PP/TP imbalance.
        capacity = min(worker.capacity_bytes for worker in workers)
        return cls(
            capacity,
            capacity * max(worker.bytes_per_token / worker.capacity_bytes for worker in workers),
            capacity * max(worker.bytes_per_state / worker.capacity_bytes for worker in workers),
        )


def capacity_from_cache_specs(specs, vllm_config, capacity_bytes: int) -> KVCapacity:
    """Derive rank-local coefficients from physical attention and recurrent cache specs."""
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    token_bytes = state_bytes = 0.0
    context_parallel = vllm_config.parallel_config.decode_context_parallel_size
    for spec in specs.values():
        if isinstance(spec, MambaSpec):
            if spec.mamba_cache_mode == "all":
                token_bytes += spec.page_size_bytes / spec.block_size
                state_bytes += spec.page_size_bytes * (1 + spec.num_speculative_blocks)
            else:
                state_bytes += spec.max_memory_usage_bytes(vllm_config)
        elif isinstance(spec, AttentionSpec):
            token_bytes += spec.page_size_bytes / (spec.block_size * context_parallel)
            # Reserve one partially filled block per independent chain.
            state_bytes += spec.page_size_bytes
        else:
            raise ValueError(f"Unsupported logical KV cache spec: {type(spec).__name__}")
    return KVCapacity(capacity_bytes, token_bytes, state_bytes)


def worker_kv_capacity(worker: Any) -> KVCapacity:
    """Inspect initialized backing allocations once, without reading GPU contents."""
    import torch

    regions: dict[int, int] = {}

    def collect(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            regions[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(worker.model_runner.kv_caches)
    if not regions:
        raise ValueError("logical KV admission requires initialized cache storage")
    return capacity_from_cache_specs(worker.model_runner.get_kv_cache_spec(), worker.vllm_config, sum(regions.values()))
