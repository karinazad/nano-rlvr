# nano-rlvr

Minimal, readable implementation of **Reinforcement Learning with Verifiable Rewards (RLVR)** — the technique behind [DeepSeek-R1](https://arxiv.org/abs/2501.12948)'s reasoning capabilities.

nanoGPT-style: single-GPU, no infra bloat, readable top-to-bottom.

## What is RLVR?

Standard RLHF trains a reward model from human preferences, then optimizes a policy against it. **RLVR skips the reward model entirely** — instead, it uses *programmatically verifiable* rewards (did the model get the right answer to a math problem?). This is:

- **Simpler** — no reward model training, no preference data collection
- **Cheaper** — reward computation is a function call, not a neural network forward pass
- **Surprisingly effective** — has been shown to elicit chain-of-thought reasoning from base models

## Algorithms

### REINFORCE (`train_reinforce.py`)

The simplest policy gradient method. For each prompt:

1. Generate one completion
2. Check if the answer is correct (reward = 0 or 1)
3. Update: `loss = -(reward - baseline) * log_prob(completion)`
4. Add KL penalty against reference model to prevent collapse

The baseline is a running mean of rewards, which reduces gradient variance.

### GRPO (`train_grpo.py`)

**Group Relative Policy Optimization** — DeepSeek-R1's algorithm. The key idea: generate *K* completions per prompt, then compute advantages *relative to the group*:

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

This eliminates the need for a critic/value network. Then optimize with PPO-style clipping:

```
ratio = pi(a|s) / pi_old(a|s)
loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

Multiple optimization epochs per batch of generated data, plus a KL penalty against the reference model.

## Quickstart

```bash
# Install with uv
uv sync

# With dev extras (wandb, ruff, pre-commit)
uv sync --all-extras
```

### Train with REINFORCE

```bash
python train_reinforce.py
```

### Train with GRPO

```bash
python train_grpo.py
```

Both scripts auto-detect GPU availability and fall back to CPU (slow but works for debugging).

## What to expect

- **Reward curves** should increase over ~100-200 steps as the model learns to produce correct answers
- **Sample completions** are printed periodically — watch for emerging step-by-step reasoning
- Arithmetic task is easier and converges faster; countdown is harder but more interesting
- On CPU with the default 0.5B model, expect ~1-5 seconds per step for REINFORCE, more for GRPO (K completions per prompt)

## Project structure

```
nano-rlvr/
├── README.md
├── pyproject.toml
├── src/
│   └── nano_rlvr/
│       ├── __init__.py
│       ├── model.py          # Load/wrap pretrained LM, batched generation
│       ├── rewards.py        # Verifiable reward: check_arithmetic, check_countdown
│       ├── data.py           # Online problem generators (no dataset files)
│       └── utils.py          # KL divergence, advantage normalization, logging
├── train_reinforce.py        # REINFORCE + verifiable rewards
└── train_grpo.py             # GRPO (DeepSeek-R1 algorithm)
```

## Design decisions

- **HuggingFace transformers for model loading** — the "nano" is about the RL algorithm, not reimplementing model infra. Default model: `Qwen/Qwen2.5-0.5B`.
- **Online problem generation** — no static datasets, no data loading pipelines.
- **Single GPU, vanilla PyTorch** — no DeepSpeed, no FSDP, no vLLM.
- **Self-contained training scripts** — each readable top-to-bottom with a config dataclass and a training loop.
- **Functional style** — no trainer classes, no callback systems. Just functions and a for-loop.

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — the GRPO algorithm and RLVR approach
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) — the original GRPO paper (DeepSeek-Math)
- [nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for the "nano" philosophy
