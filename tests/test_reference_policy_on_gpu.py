# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real FSDP2 smoke: RUN_REFERENCE_GPU_TESTS=1 pytest -q this_file."""

import asyncio
import os
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding

from laminar.train.trainer import AsyncRLTrainerConfig
from laminar.workers.fsdp_workers import AsyncActorWorker
from test_async_checkpoint_on_cpu import _trainer
from test_reference_policy_on_cpu import _batch
from test_train_optimizer_defaults_on_cpu import _compose_train

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REFERENCE_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="requires opt-in and a free CUDA GPU",
)


def _tiny_model(path):
    path.mkdir()
    torch.manual_seed(7)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=1,
        )
    ).save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0")),
        unk_token="0",
        pad_token="0",
        eos_token="1",
    )
    tokenizer.save_pretrained(path)


class _SingleRankGroup:
    def __init__(self, worker):
        self.worker = worker

    def get_reference_policy_identity(self):
        return [self.worker.get_reference_policy_identity()]

    def update_actor(self, td):
        return asyncio.run(self.worker.update_actor(td))

    def __getattr__(self, name):
        return getattr(self.worker, name)


def test_fsdp2_reference_three_updates_and_checkpoint_restore(tmp_path, monkeypatch):
    for key, value in {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "0",
    }.items():
        monkeypatch.setenv(key, value)
    model_path = tmp_path / "base"
    _tiny_model(model_path)
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"file://{tmp_path / 'rendezvous'}",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", 0),
    )
    workers = []
    try:
        config = _compose_train(
            "model_engine=dp",
            f"actor_rollout_ref.model.path={model_path}",
            "actor_rollout_ref.model.use_remove_padding=False",
            "+actor_rollout_ref.model.override_config.attn_implementation=eager",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.use_torch_compile=False",
            "actor_rollout_ref.ref.use_torch_compile=False",
            "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            "actor_rollout_ref.actor.optim.lr=0.001",
            "actor_rollout_ref.actor.optim.total_training_steps=4",
            "actor_rollout_ref.rollout.n=2",
            "data.train_batch_size=1",
        )
        OmegaConf.resolve(config)
        worker = AsyncActorWorker(config=deepcopy(config.actor_rollout_ref), role="actor_ref")
        workers.append(worker)
        worker.init_model()
        group = _SingleRankGroup(worker)
        identity = group.get_reference_policy_identity()
        assert all(not p.requires_grad and p.grad is None for p in worker.ref.engine.module.parameters())
        assert all(p.to_local().device.type == "cpu" for p in worker.ref.engine.module.parameters())

        trainer = _trainer(tmp_path / "checkpoints", actor_wg=group, relay_version=3)
        config.trainer.default_local_dir = str(tmp_path / "checkpoints")
        trainer.config = config
        trainer.trainer_config = AsyncRLTrainerConfig()
        trainer._reference_policy_identity = identity
        initial_actor = [p.to_local().detach().cpu().clone() for p in worker.actor.engine.module.parameters()]
        for step in range(1, 4):
            trainer.global_steps = step
            td = _batch()
            td["advantages"] = td["response_mask"].float()
            tu.assign_non_tensor(td, global_token_num=[5, 4], temperature=1.0)
            td = left_right_2_no_padding(td)
            metrics = {}
            trainer._compute_actor_update_inputs(td, metrics)
            trainer._update_actor(td, metrics)
            assert metrics["actor/update/optimizer_updates_completed"] == 1
            assert metrics["actor/update/optimizer_updates_skipped"] == 0
            assert torch.isfinite(td["ref_log_prob"]).all()
            assert group.get_reference_policy_identity() == identity
            assert all(p.to_local().device.type == "cpu" for p in worker.ref.engine.module.parameters())
        updated_actor = [p.to_local().detach().cpu().clone() for p in worker.actor.engine.module.parameters()]
        assert any(not torch.equal(before, after) for before, after in zip(initial_actor, updated_actor, strict=True))

        asyncio.run(trainer._save_checkpoint())
        restored = AsyncActorWorker(config=deepcopy(config.actor_rollout_ref), role="actor_ref")
        workers.append(restored)
        restored.init_model()
        trainer.actor_wg = _SingleRankGroup(restored)
        trainer.config.trainer.resume_mode = "resume_path"
        trainer.config.trainer.resume_from_path = str(tmp_path / "checkpoints/global_step_3")
        assert asyncio.run(trainer._load_checkpoint())
        assert trainer.actor_wg.get_reference_policy_identity() == identity
        for expected, actual in zip(updated_actor, restored.actor.engine.module.parameters(), strict=True):
            torch.testing.assert_close(actual.to_local().cpu(), expected)
    finally:
        for worker in workers:
            worker.send_loop.call_soon_threadsafe(worker.send_loop.stop)
            worker.send_thread.join()
            worker.send_loop.close()
        torch.distributed.destroy_process_group()
