# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Same-version Repack policy protocol, state views, and default BFD planner.

The Router owns replica eligibility, version grouping, invariant validation, and
atomic route migration; a ``RepackPolicy`` only maps destinations to source
server ids within one eligible same-version cohort.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

from omegaconf import OmegaConf


@dataclass(frozen=True)
class CommittedRepackGroup:
    """One destination group already atomically committed by the Router."""

    sources: tuple[str, ...]
    destination: str
    weight_version: int


@dataclass(frozen=True)
class CommittedAbortSource:
    """A busy stale server already set draining with no destination (latest-version continuation)."""

    server_id: str
    weight_version: int


@dataclass(frozen=True)
class RepackPlanResult:
    """Minimal Router plan-and-commit result consumed by the lifecycle manager."""

    groups: tuple[CommittedRepackGroup, ...]
    idle_sources: tuple[str, ...]
    abort_sources: tuple[CommittedAbortSource, ...] = ()


@dataclass(frozen=True)
class RepackSession:
    """Read-only snapshot of one sticky session bound to a candidate replica.

    ``context_tokens`` reflects the most recent request lease on this session
    (``0`` before its first turn); ``inflight_leases > 0`` indicates an active
    generation phase.
    """

    session_id: str
    context_tokens: int = 0
    inflight_leases: int = 0


@dataclass(frozen=True)
class RepackReplica:
    """Read-only state of one active, runtime-available stale replica."""

    server_id: str
    weight_version: int
    kv_cache_usage: Any | None
    waiting_requests: int = 0
    running_requests: int = 0
    inflight_leases: int = 0
    sessions: tuple[RepackSession, ...] = ()


@runtime_checkable
class RepackPolicy(Protocol):
    """Strategy interface mapping ``destination_id -> source_ids`` in one cohort."""

    def plan(self, replicas: Sequence[RepackReplica]) -> Mapping[str, Sequence[str]]: ...


def _valid_kv_cache_usage(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


@dataclass(frozen=True)
class BestFitRepackPolicy:
    """Default same-version Best-Fit Decreasing Repack policy bounded by backend KV usage."""

    source_max_kv_cache_usage: float = 0.5
    destination_max_kv_cache_usage: float = 0.95

    def __post_init__(self) -> None:
        for name, value in (
            ("source_max_kv_cache_usage", self.source_max_kv_cache_usage),
            ("destination_max_kv_cache_usage", self.destination_max_kv_cache_usage),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
            object.__setattr__(self, name, float(value))

    def plan(self, replicas: Sequence[RepackReplica]) -> dict[str, tuple[str, ...]]:
        validated_kv: dict[str, float] = {}
        for replica in replicas:
            usage = _valid_kv_cache_usage(replica.kv_cache_usage)
            if usage is None:
                raise ValueError(
                    f"server {replica.server_id!r} has missing or invalid kv_cache_usage for repack planning"
                )
            validated_kv[replica.server_id] = usage

        if len(replicas) < 2:
            return {}
        source_candidates = sorted(
            (
                (validated_kv[replica.server_id], replica.server_id)
                for replica in replicas
                if replica.sessions and validated_kv[replica.server_id] <= self.source_max_kv_cache_usage
            ),
            key=lambda candidate: (-candidate[0], candidate[1]),
        )
        if not source_candidates:
            return {}

        planned_sources: dict[str, list[str]] = {}
        planned_destination_kv = dict(validated_kv)
        for source_kv, source_id in source_candidates:
            if source_id not in planned_destination_kv or source_id in planned_sources:
                continue
            best_fit: tuple[float, str] | None = None
            for destination_id, destination_kv in planned_destination_kv.items():
                if destination_id == source_id or destination_kv > self.destination_max_kv_cache_usage:
                    continue
                projected_kv = destination_kv + source_kv
                if projected_kv > self.destination_max_kv_cache_usage:
                    continue
                candidate = (projected_kv, destination_id)
                if (
                    best_fit is None
                    or candidate[0] > best_fit[0]
                    or (candidate[0] == best_fit[0] and candidate[1] < best_fit[1])
                ):
                    best_fit = candidate
            if best_fit is None:
                continue
            projected_kv, destination_id = best_fit
            del planned_destination_kv[source_id]
            planned_sources.setdefault(destination_id, []).append(source_id)
            planned_destination_kv[destination_id] = projected_kv

        return {dst: tuple(planned_sources[dst]) for dst in sorted(planned_sources)}


def resolve_repack_policy(policy: Any) -> RepackPolicy:
    """Resolve an optional ``RepackPolicy`` instance or ``_target_`` mapping."""
    if OmegaConf.is_config(policy):
        policy = OmegaConf.to_container(policy, resolve=True)
    if policy is None:
        return BestFitRepackPolicy()
    if isinstance(policy, Mapping):
        spec = dict(policy)
        target = spec.pop("_target_", None)
        if not isinstance(target, str) or "." not in target:
            raise ValueError("repack policy mapping must specify a dotted '_target_' class or callable")
        module_name, attr_name = target.rsplit(".", 1)
        policy = getattr(import_module(module_name), attr_name)(**spec)
    if not isinstance(policy, RepackPolicy):
        raise TypeError(f"repack policy must implement plan(replicas), got {type(policy)!r}")
    return policy


def _commit_repack_groups(
    replicas: Sequence[RepackReplica],
    plan: Mapping[str, Sequence[str]],
) -> list[CommittedRepackGroup]:
    """Validate a ``destination -> sources`` plan before Router state mutation."""
    if not isinstance(plan, Mapping):
        raise TypeError(f"repack policy must return a destination -> sources mapping, got {type(plan)!r}")
    if not plan:
        return []
    by_id = {replica.server_id: replica for replica in replicas}
    version = replicas[0].weight_version
    claimed_replicas: set[str] = set()
    committed: list[CommittedRepackGroup] = []
    for destination, raw_sources in plan.items():
        if isinstance(raw_sources, str | bytes) or not isinstance(raw_sources, Iterable):
            raise TypeError(f"repack sources for {destination!r} must be an iterable of server_ids")
        sources = tuple(raw_sources)
        if not sources:
            raise ValueError(f"repack group for destination {destination!r} has empty sources")
        if destination not in by_id:
            raise ValueError(f"repack destination {destination!r} is not in the eligible version cohort")
        if len(set(sources)) != len(sources):
            raise ValueError(f"repack group for destination {destination!r} contains duplicate sources")
        if destination in sources:
            raise ValueError(f"repack replica {destination!r} cannot be both source and destination")
        for source_id in sources:
            source = by_id.get(source_id)
            if source is None:
                raise ValueError(f"repack source {source_id!r} is not in the eligible version cohort")
            if not source.sessions:
                raise ValueError(f"repack source {source_id!r} has no bound sessions to migrate")
        group_members = {destination, *sources}
        overlap = claimed_replicas & group_members
        if overlap:
            raise ValueError(f"repack groups overlap on replicas: {sorted(overlap)}")
        claimed_replicas.update(group_members)
        committed.append(
            CommittedRepackGroup(
                sources=sources,
                destination=destination,
                weight_version=version,
            )
        )
    return committed
