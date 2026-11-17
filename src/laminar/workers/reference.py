# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Fixed reference-policy configuration and ownership for async actor workers."""

from __future__ import annotations

import hashlib

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass


def validate_reference_policy_config(config: DictConfig) -> None:
    if OmegaConf.select(config, "algorithm.use_kl_in_reward", default=False):
        raise ValueError("async_rl does not support KL in reward; use actor.use_kl_loss")
    if not OmegaConf.select(config, "actor_rollout_ref.actor.use_kl_loss", default=False):
        return
    settings = config.actor_rollout_ref
    if (
        settings.model.get("lora_rank", 0) > 0
        or settings.model.get("lora", {}).get("rank", 0) > 0
        or settings.model.get("lora_adapter_path") is not None
    ):
        raise ValueError("async reference-policy KL supports full-model actors only")
    if settings.ref.strategy != settings.actor.strategy:
        raise ValueError("async actor/ref strategies must match")
    megatron = settings.actor.strategy == "megatron"
    engine_key = "megatron" if megatron else "fsdp_config"
    actor = omega_conf_to_dataclass(settings.actor[engine_key])
    ref = omega_conf_to_dataclass(settings.ref[engine_key])
    parallel_fields = (
        (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "virtual_pipeline_model_parallel_size",
            "context_parallel_size",
            "dynamic_context_parallel",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "sequence_parallel",
            "use_mbridge",
        )
        if megatron
        else ("fsdp_size", "ulysses_sequence_parallel_size")
    )
    for field in parallel_fields:
        actor_value, ref_value = getattr(actor, field), getattr(ref, field)
        if field == "ulysses_sequence_parallel_size":
            if settings.actor.get(field, 1) > 1:
                actor_value = settings.actor[field]
            if settings.ref.get(field, 1) > 1:
                ref_value = settings.ref[field]
        elif field == "expert_tensor_parallel_size":
            actor_value = actor.tensor_model_parallel_size if actor_value is None else actor_value
            ref_value = ref.tensor_model_parallel_size if ref_value is None else ref_value
        elif field == "fsdp_size":
            world_size = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
            actor_value = world_size if actor_value < 0 or actor_value >= world_size else actor_value
            ref_value = world_size if ref_value < 0 or ref_value >= world_size else ref_value
        if actor_value != ref_value:
            raise ValueError(f"async actor/ref {field} must match: actor={actor_value!r}, ref={ref_value!r}")
    if not ref.forward_only:
        raise ValueError("async reference engine must be forward_only")
    # Pinned verl FSDP forces CPU offload for forward-only models.
    if megatron and config.get("training_strategy") == "colocate_async" and not ref.param_offload:
        raise ValueError("colocate_async requires reference parameter offload")


def _engine_modules(engine) -> list[torch.nn.Module]:
    return list(engine.module) if isinstance(engine.module, list | tuple) else [engine.module]


def _local_parameter(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


class ReferencePolicyMixin:
    """Add the actor_ref role without creating an in-worker rollout engine."""

    def __init__(self, *, config: DictConfig, role: str, **kwargs):
        # The public verl constructor admits actor and actor_rollout_ref only;
        # its init_model already constructs both models by role membership.
        super().__init__(config=config, role="actor" if role == "actor_ref" else role, **kwargs)
        if role == "actor_ref":
            self.role = role
            self._is_ref = True

    def _freeze_reference_policy(self) -> None:
        if not self._is_ref:
            return
        if not self.ref.engine.engine_config.forward_only or self.ref.engine.optimizer is not None:
            raise RuntimeError("reference policy must be forward-only without an optimizer")
        actor_storage = {
            (local.device, local.untyped_storage().data_ptr())
            for module in _engine_modules(self.actor.engine)
            for parameter in module.parameters()
            if (local := _local_parameter(parameter)).numel()
        }
        for module in _engine_modules(self.ref.engine):
            for parameter in module.parameters():
                local = _local_parameter(parameter)
                if local.numel() and (local.device, local.untyped_storage().data_ptr()) in actor_storage:
                    raise RuntimeError("actor and reference policy must own independent parameter storage")
            module.requires_grad_(False)
            module.eval()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def get_reference_policy_identity(self) -> dict:
        """Sample each local shard for resume identity; this is not a full-weight checksum."""
        if not self._is_ref:
            raise RuntimeError("reference policy is not enabled")
        digest = hashlib.sha256()
        for index, module in enumerate(_engine_modules(self.ref.engine)):
            for name, parameter in module.named_parameters():
                if parameter.requires_grad or parameter.grad is not None:
                    raise RuntimeError("reference parameters must remain frozen without gradients")
                local = _local_parameter(parameter).detach().reshape(-1)
                digest.update(f"{index}:{name}:{parameter.shape}:{parameter.dtype}".encode())
                sample = torch.cat((local[:64], local[-64:])).cpu().contiguous()
                digest.update(sample.view(torch.uint8).numpy().tobytes())
        return {
            "rank": self.rank,
            "model_path": str(self.config.model.path),
            "sampled_parameter_sha256": digest.hexdigest(),
        }
