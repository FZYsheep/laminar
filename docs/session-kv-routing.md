# Per-replica session budgets

Set `rollout_router.logical_kv.enabled=True` to enable per-replica admission and
logical KV Repack. The default is `False`. This policy supplements the total
session limit and replaces snapshot-based admission and Repack thresholds.

The Router sums demand across all bound sessions, including sessions waiting on
tools:

```text
logical_kv_bytes =
    bytes_per_token * (context_tokens + reserved_tokens)
    + bytes_per_state * state_count
```

Capacity and coefficients come from initialized vLLM worker storage and cache
specs. Shared prefixes are counted per context; logical demand is an estimate,
not measured GPU occupancy.

Settings under `rollout_router.logical_kv`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `max_sessions_per_replica` | `24` | New-session cap |
| `admission_capacity_multiplier` | `1.2` | New-session demand / capacity limit |
| `growth_tokens_per_state` | `512` | Reserved tokens per context |
| `repack_capacity_multiplier` | `1.0` | Combined Repack demand / capacity limit |
| `repack_session_multiplier` | `1.5` | Repack session allowance: 36 by default |
| `repack_max_waiting_requests` | `32` | Combined waiting-request / lease bound |

Retries do not consume another slot. Admitted sessions can finish after growth
exceeds the budget; growth blocks new admissions and further consolidation.
`release_route` frees the session budget. Repack merges only same-version
replicas, checks cumulative KV/session/request limits, and preserves source leases.

The client updates single-context budgets from prompt/output lengths. Owners of
multiple contexts must publish absolute `context_tokens` and `state_count` through
`update_session_budget(..., revision=...)` and pass `budget_managed=True` to
generation. Revisions increase per session; stale updates are ignored. The pinned
Uni-Agent does not automatically publish these totals.

Capacity assumes a stable replica allocation; changes after backend recovery
are not requalified.
