# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPActorConfig
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding

from laminar.train.trainer import AsyncRLTrainer
from laminar.workers.fsdp_workers import AsyncActorWorker
from laminar.workers.reference import validate_reference_policy_config
from test_async_checkpoint_on_cpu import _trainer, _write_minimal_checkpoint
from test_train_optimizer_defaults_on_cpu import _compose_train


def _config(engine="dp"):
    config = _compose_train(f"model_engine={engine}")
    OmegaConf.set_struct(config, False)
    if engine == "megatron":
        config.actor_rollout_ref.actor.strategy = "megatron"
    OmegaConf.resolve(config)
    config.actor_rollout_ref.actor.use_kl_loss = True
    return config


@pytest.mark.parametrize("engine", ["dp", "megatron"])
def test_reference_config_uses_effective_defaults(engine):
    config = _config(engine)
    settings = config.actor_rollout_ref
    key = "megatron" if engine == "megatron" else "fsdp_config"
    # An omitted False and an explicit False are the same effective config.
    field = "dynamic_context_parallel" if engine == "megatron" else "forward_prefetch"
    settings.actor[key].pop(field)
    settings.ref[key][field] = False
    validate_reference_policy_config(config)


@pytest.mark.parametrize(
    ("engine", "field", "value"),
    [
        ("dp", "fsdp_size", 1),
        ("dp", "ulysses_sequence_parallel_size", 2),
        ("megatron", "tensor_model_parallel_size", 2),
        ("megatron", "pipeline_model_parallel_size", 2),
        ("megatron", "context_parallel_size", 2),
        ("megatron", "expert_model_parallel_size", 2),
        ("megatron", "dynamic_context_parallel", True),
    ],
)
def test_reference_config_rejects_parallel_mismatch(engine, field, value):
    config = _config(engine)
    key = "megatron" if engine == "megatron" else "fsdp_config"
    config.actor_rollout_ref.ref[key][field] = value
    if field == "dynamic_context_parallel":
        config.actor_rollout_ref.ref.megatron.max_seqlen_per_dp_cp_rank = 2048
    with pytest.raises(ValueError, match=field):
        validate_reference_policy_config(config)


@pytest.mark.parametrize("engine", ["dp", "megatron"])
def test_colocate_requires_reference_offload(engine):
    config = _config(engine)
    config.training_strategy = "colocate_async"
    key = "megatron" if engine == "megatron" else "fsdp_config"
    config.actor_rollout_ref.ref[key].param_offload = False
    if engine == "megatron":
        with pytest.raises(ValueError, match="reference parameter offload"):
            validate_reference_policy_config(config)
    else:
        # Native FSDP forward-only CPU offload does not depend on param_offload.
        validate_reference_policy_config(config)
    config.actor_rollout_ref.ref[key].param_offload = True
    validate_reference_policy_config(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model.lora_rank", 8, "full-model"),
        ("model.lora.rank", 8, "full-model"),
        ("model.lora_adapter_path", "/adapter", "full-model"),
        ("ref.strategy", "megatron", "strategies"),
        ("ref.fsdp_config.forward_only", False, "forward_only"),
    ],
)
def test_reference_config_rejects_unsupported_modes(path, value, message):
    config = _config()
    OmegaConf.update(config.actor_rollout_ref, path, value)
    with pytest.raises(ValueError, match=message):
        validate_reference_policy_config(config)


@pytest.mark.parametrize("engine", ["fsdp", "megatron"])
@pytest.mark.parametrize("role", ["actor", "actor_ref"])
def test_async_role_adapts_public_verl_without_rollout(monkeypatch, engine, role):
    worker_cls = AsyncActorWorker
    if engine == "megatron":
        pytest.importorskip("megatron.core")
        from laminar.workers.megatron_workers import AsyncMegatronActorWorker

        worker_cls = AsyncMegatronActorWorker
    calls = []

    def base_init(self, *, config, role):
        calls.append(role)
        self.config = config
        self.role = role
        self._is_actor, self._is_rollout, self._is_ref = True, False, False

    monkeypatch.setattr(ActorRolloutRefWorker, "__init__", base_init)
    worker = worker_cls(config=OmegaConf.create({}), role=role)
    try:
        assert calls == ["actor"]
        assert worker.role == role
        assert worker._is_actor and not worker._is_rollout
        assert worker._is_ref is (role == "actor_ref")
    finally:
        if worker_cls is AsyncActorWorker:
            worker.send_loop.call_soon_threadsafe(worker.send_loop.stop)
            worker.send_thread.join()
            worker.send_loop.close()


def _reference_worker(*, shared=False, chunks=False):
    worker = object.__new__(AsyncActorWorker)
    worker._is_ref = True
    worker._rank = 0
    worker.config = OmegaConf.create({"model": {"path": "/immutable/base"}})
    actor = torch.nn.Linear(4, 4)
    ref = actor if shared else deepcopy(actor)
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=actor))
    worker.ref = SimpleNamespace(
        engine=SimpleNamespace(
            module=[ref] if chunks else ref,
            engine_config=SimpleNamespace(forward_only=True),
            optimizer=None,
        )
    )
    return worker, actor, ref


@pytest.mark.parametrize("chunks", [False, True])
def test_reference_freezing_preserves_actor_and_detects_changed_identity(chunks):
    worker, actor, ref = _reference_worker(chunks=chunks)
    worker._freeze_reference_policy()
    identity = worker.get_reference_policy_identity()
    assert all(parameter.requires_grad for parameter in actor.parameters())
    assert not ref.training and all(not parameter.requires_grad for parameter in ref.parameters())
    with torch.no_grad():
        actor.weight.add_(1)
    assert worker.get_reference_policy_identity() == identity
    with torch.no_grad():
        ref.weight.add_(1)
    assert worker.get_reference_policy_identity() != identity


def test_reference_rejects_shared_storage_and_optimizer():
    worker, _, _ = _reference_worker(shared=True)
    with pytest.raises(RuntimeError, match="independent parameter storage"):
        worker._freeze_reference_policy()
    worker, _, _ = _reference_worker()
    worker.ref.engine.optimizer = object()
    with pytest.raises(RuntimeError, match="without an optimizer"):
        worker._freeze_reference_policy()


def _batch():
    return TensorDict(
        {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 0, 0]]),
            "prompts": torch.tensor([[0, 1, 2], [1, 2, 3]]),
            "responses": torch.tensor([[3, 4, 5], [4, 0, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 0, 0]]),
            "response_mask": torch.tensor([[1, 0, 1], [1, 0, 0]]),
        },
        batch_size=[2],
    )


@pytest.mark.parametrize("nested", [False, True])
def test_reference_forward_aligns_actions_and_masks_non_action_nan(nested):
    trainer = object.__new__(AsyncRLTrainer)
    td = _batch()
    if nested:
        td = left_right_2_no_padding(td)
    values = [torch.tensor([9.0, -1.0, float("nan"), -2.0, 9.0]), torch.tensor([9.0, 9.0, -3.0, 9.0])]

    def compute(infer_td):
        assert infer_td["input_ids"].is_nested
        assert not tu.get(infer_td, "calculate_entropy")
        assert not tu.get(infer_td, "compute_loss")
        assert not tu.get(infer_td, "disable_auto_offload")
        return TensorDict({"log_probs": torch.nested.as_nested_tensor(values, layout=torch.jagged)}, [2])

    trainer.actor_wg = SimpleNamespace(compute_ref_log_prob=compute)
    metrics = {}
    trainer._compute_ref_log_prob(td, metrics)
    torch.testing.assert_close(td["ref_log_prob"], torch.tensor([[-1.0, 0.0, -2.0], [-3.0, 0.0, 0.0]]))
    assert metrics["timing/ref_log_prob"] >= 0


def test_reference_forward_rejects_missing_and_nonfinite_action_output():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.actor_wg = SimpleNamespace(compute_ref_log_prob=lambda _: TensorDict({}, [2]))
    with pytest.raises(RuntimeError, match="did not return"):
        trainer._compute_ref_log_prob(_batch(), {})
    trainer.actor_wg.compute_ref_log_prob = lambda _: TensorDict(
        {
            "log_probs": torch.nested.as_nested_tensor(
                [torch.full((5,), float("nan")), torch.zeros(4)], layout=torch.jagged
            )
        },
        [2],
    )
    with pytest.raises(RuntimeError, match="finite on action"):
        trainer._compute_ref_log_prob(_batch(), {})


@pytest.mark.parametrize("enabled", [False, True])
def test_reference_switch_controls_forward_in_update_preparation(enabled):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"actor": {"use_kl_loss": enabled}}})
    trainer._compute_old_log_probs = Mock()
    trainer._compute_ref_log_prob = Mock()
    trainer._apply_rollout_correction = Mock()
    trainer._compute_actor_update_inputs(TensorDict({}, [2]), {})
    assert trainer._compute_ref_log_prob.call_count == int(enabled)


def test_native_ppo_kl_coefficient_and_non_action_tokens():
    td = _batch()
    td["old_log_probs"] = torch.full((2, 3), -0.4)
    td["advantages"] = torch.ones(2, 3) * td["response_mask"]
    td["ref_log_prob"] = torch.full((2, 3), -0.8)
    tu.assign_non_tensor(td, dp_size=1, batch_num_tokens=3, global_batch_size=2)
    td = left_right_2_no_padding(td)
    config = FSDPActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        use_kl_loss=True,
        kl_loss_type="mse",
        global_batch_info={},
        kl_loss_coef=0.0,
    )
    outputs = {
        "log_probs": torch.nested.as_nested_tensor(
            [torch.full((5,), -0.4), torch.full((4,), -0.4)], layout=torch.jagged
        )
    }
    baseline, _ = ppo_loss(config, outputs, td)
    config = replace(config, kl_loss_coef=0.2)
    loss, metrics = ppo_loss(config, outputs, td)
    assert float(loss - baseline) == pytest.approx(metrics["kl_loss"].aggregate() * 0.2, abs=1e-7)
    changed = td.clone()
    changed["ref_log_prob"] = changed["ref_log_prob"].masked_fill(~changed["response_mask"].bool(), -1000)
    masked_loss, _ = ppo_loss(config, outputs, changed)
    torch.testing.assert_close(masked_loss, loss)


@pytest.mark.parametrize("engine", ["dp", "megatron"])
def test_reference_accepts_explicit_backend_defaults(engine):
    config = _config(engine)
    if engine == "dp":
        config.actor_rollout_ref.ref.fsdp_config.fsdp_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    else:
        config.actor_rollout_ref.ref.megatron.expert_tensor_parallel_size = 1
    validate_reference_policy_config(config)


def _enable_checkpoint_reference(trainer):
    trainer.config.actor_rollout_ref.actor.use_kl_loss = True
    trainer._reference_policy_identity = [{"rank": 0, "sampled_parameter_sha256": "base"}]
    trainer.actor_wg.get_reference_policy_identity = lambda: trainer._reference_policy_identity


def test_reference_checkpoint_save_restore_and_mismatch_fail_before_actor_load(tmp_path):
    trainer = _trainer(tmp_path)
    _enable_checkpoint_reference(trainer)
    asyncio.run(trainer._save_checkpoint())
    checkpoint_dir = tmp_path / "global_step_12"
    assert json.loads((checkpoint_dir / "reference_policy.json").read_text()) == trainer._reference_policy_identity
    assert asyncio.run(trainer._load_checkpoint())
    trainer.actor_wg.load_calls.clear()
    trainer.actor_wg.get_reference_policy_identity = lambda: [{"rank": 0, "sampled_parameter_sha256": "changed"}]
    with pytest.raises(RuntimeError, match="does not match"):
        asyncio.run(trainer._load_checkpoint())
    assert not trainer.actor_wg.load_calls


def test_reference_checkpoint_requires_identity_and_same_objective(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint_dir = _write_minimal_checkpoint(tmp_path)
    _enable_checkpoint_reference(trainer)
    with pytest.raises(RuntimeError, match="requires checkpoint"):
        asyncio.run(trainer._load_checkpoint())
    (checkpoint_dir / "reference_policy.json").write_text(json.dumps(trainer._reference_policy_identity))
    trainer.config.actor_rollout_ref.actor.use_kl_loss = False
    with pytest.raises(RuntimeError, match="use_kl_loss=True"):
        asyncio.run(trainer._load_checkpoint())
    assert not trainer.actor_wg.load_calls


def test_changed_reference_cannot_commit_checkpoint_marker(tmp_path):
    trainer = _trainer(tmp_path)
    _enable_checkpoint_reference(trainer)
    trainer.actor_wg.get_reference_policy_identity = lambda: []
    with pytest.raises(RuntimeError, match="changed since initialization"):
        asyncio.run(trainer._save_checkpoint())
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
