# Score centering

Laminar carries vLLM sampler heads through continuation, Uni-Agent trajectories,
and TransferQueue batches into verl's native score-centering loss. It uses the
existing head/tail estimator and bypass loss; old-policy recomputation and the
separate importance-sampling pass are skipped.

Use random vLLM sampling, positive `topk_log_probs`, `use_remove_padding=True`,
and `use_fused_kernels=False`. Bucket padding, Dynamic CP, distillation, and
tree trajectories are unsupported.

After applying the Qwen3.5-4B SWE recipe PR, run:

```bash
bash recipes/swe/train_qwen35_4b_score_centering.sh
```

It keeps the recipe's 4 trainer + 4 rollout GPUs, disables KL, and collects
32 sampler heads per generated token. Set model, data, and sandbox variables
as described in `recipes/swe/README.md`.
