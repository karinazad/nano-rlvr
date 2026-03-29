# nano-rlvr

Minimal implementation of Reinforcement Learning with Verifiable Rewards (RLVR) — used e.g in [DeepSeek-R1](https://arxiv.org/abs/2501.12948). nanoGPT-style: single GPU, minimal infra, and readable top-to-bottom.


## Algorithms

### REINFORCE (`train_reinforce.py`)

Simplest policy gradient. For each prompt:

1. Generate a completion
2. Check the answer (reward = 0 or 1)
3. `loss = -(reward - baseline) * log_prob(completion)`
4. KL penalty against a frozen reference model to prevent collapse

Baseline is a running mean of rewards.

### GRPO (`train_grpo.py`)

Group Relative Policy Optimization — the DeepSeek-R1 algorithm. For each prompt, generate K completions and compute advantages relative to the group:

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

No critic network needed. Then PPO-style clipped optimization:

```
ratio = pi(a|s) / pi_old(a|s)
loss = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)
```

Multiple epochs per batch, plus KL penalty.

## Quickstart

```bash
uv sync
python train_reinforce.py
python train_grpo.py
```

Dev extras (wandb, ruff, pre-commit): `uv sync --all-extras`

Both scripts fall back to CPU if CUDA isn't available.

## What to expect

Reward should climb over ~100-200 steps. Sample completions are printed periodically — you can watch the model start producing step-by-step reasoning. Arithmetic converges faster; countdown is harder.

## Structure

```
src/nano_rlvr/
  model.py      Model loading, batched generation with log-probs
  rewards.py    Verifiable reward functions (arithmetic, countdown)
  data.py       Online problem generators
  utils.py      KL divergence, advantage normalization, logging

train_reinforce.py   REINFORCE + verifiable rewards
train_grpo.py        GRPO (DeepSeek-R1 algorithm)
```

The "nano" is about the RL algorithm, not the model — we use HuggingFace transformers to load `Qwen/Qwen2.5-0.5B`. No DeepSpeed, no FSDP, no vLLM. No trainer classes or callback systems. Just functions and a for-loop.

## References

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO + RLVR for reasoning
- [DeepSeek-Math](https://arxiv.org/abs/2402.03300) — original GRPO paper
- [nanoGPT](https://github.com/karpathy/nanoGPT)
