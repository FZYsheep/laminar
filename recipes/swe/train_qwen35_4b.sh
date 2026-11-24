#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Qwen3.5-4B: four FSDP2 Actor/Ref GPUs and four TP1 vLLM replicas.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ID=${MODEL_ID:-Qwen3.5-4B}
export MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME:-${HOME}/verl}/models/Qwen3.5-4B"}
export EXP_NAME=${EXP_NAME:-qwen35-4b-swe}
export TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0
export LAMINAR_RAY_ENV_VARS="$(
    python3 -c 'import json, os; env = json.loads(os.environ.get("LAMINAR_RAY_ENV_VARS", "{}")); env["TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"] = "0"; print(json.dumps(env))'
)"

exec bash "${SCRIPT_DIR}/train.sh" \
    model_engine=dp \
    data.train_batch_size=4 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_seqs=96 \
    actor_rollout_ref.rollout.max_model_len=24576 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=0 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=4 \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=16 \
    rollout_router.logical_kv.enabled=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_training_steps=3 \
    trainer.weight_publish_interval=1 \
    trainer.save_freq=3 \
    relay.backend=ucx \
    "$@"
