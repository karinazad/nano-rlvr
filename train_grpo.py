"""Group Relative Policy Optimization (GRPO) with Verifiable Rewards.

This is the algorithm behind DeepSeek-R1. For each prompt, generate K
completions (a "group"), compute advantages *relative to the group* (no
critic needed), and optimize with PPO-style clipped objective.

Usage:
    python train_grpo.py
"""

import copy
from dataclasses import dataclass

import torch

from nano_rlvr.data import generate_arithmetic_problems, generate_countdown_problems
from nano_rlvr.model import generate_completions, get_per_token_logps, load_model
from nano_rlvr.rewards import check_arithmetic, check_countdown
from nano_rlvr.utils import compute_kl_divergence, set_seed, setup_logging


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda"
    task: str = "arithmetic"  # "arithmetic" or "countdown"
    lr: float = 1e-6
    batch_size: int = 4  # number of prompts per step
    group_size: int = 8  # completions per prompt (K)
    max_steps: int = 200
    max_new_tokens: int = 256
    temperature: float = 0.7
    clip_epsilon: float = 0.2  # PPO clipping
    kl_coef: float = 0.05  # KL penalty coefficient
    num_epochs_per_batch: int = 2  # PPO-style mini-epochs
    seed: int = 42
    use_wandb: bool = False
    log_interval: int = 1
    sample_interval: int = 10


def train(cfg: Config):
    set_seed(cfg.seed)

    if not torch.cuda.is_available() and cfg.device == "cuda":
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    log = setup_logging(use_wandb=cfg.use_wandb, run_name="grpo")

    # Load model and create frozen reference copy
    print(f"Loading model {cfg.model_name}...")
    model, tokenizer = load_model(cfg.model_name, cfg.device)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Pick task
    if cfg.task == "arithmetic":
        generate_fn = generate_arithmetic_problems
    elif cfg.task == "countdown":
        generate_fn = generate_countdown_problems
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    for step in range(1, cfg.max_steps + 1):
        model.train()

        # 1) Sample problems
        problems = generate_fn(cfg.batch_size)
        prompts = [p for p, _ in problems]
        answers = [a for _, a in problems]

        # 2) Generate K completions per prompt
        gen = generate_completions(
            model,
            tokenizer,
            prompts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            num_samples=cfg.group_size,
        )
        # gen["completions"] has batch_size * group_size entries
        # Ordering: [prompt0_sample0, prompt0_sample1, ..., prompt1_sample0, ...]

        # 3) Compute rewards for each completion
        all_rewards = []
        for i, completion in enumerate(gen["completions"]):
            prompt_idx = i // cfg.group_size
            if cfg.task == "arithmetic":
                r = check_arithmetic(completion, answers[prompt_idx])
            else:
                nums_str = prompts[prompt_idx].split("[")[1].split("]")[0]
                numbers = [int(n.strip()) for n in nums_str.split(",")]
                r = check_countdown(completion, numbers, answers[prompt_idx])
            all_rewards.append(r)

        rewards_t = torch.tensor(all_rewards, device=cfg.device)
        mean_reward = rewards_t.mean().item()

        # 4) Compute group-relative advantages
        # Reshape to (batch_size, group_size), normalize within each group
        rewards_grouped = rewards_t.view(cfg.batch_size, cfg.group_size)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = (rewards_grouped - group_mean) / (group_std + 1e-8)
        # Handle groups where all rewards are identical (std=0)
        all_same = (group_std < 1e-8).squeeze(-1)
        advantages[all_same] = 0.0
        # Flatten back to (batch_size * group_size,)
        advantages = advantages.view(-1)

        # 5) Store old log-probs (frozen, from generation)
        old_logps = gen["logps"].detach()  # (B*K, comp_len)
        comp_mask = gen["completion_mask"]
        full_ids = gen["full_ids"]
        full_mask = gen["full_mask"]
        prompt_len = gen["prompt_ids"].shape[1]

        min_len = min(old_logps.shape[1], comp_mask.shape[1])
        old_logps = old_logps[:, :min_len]
        comp_mask = comp_mask[:, :min_len]

        # Per-sequence old log-probs
        old_seq_logps = (old_logps * comp_mask).sum(dim=1)

        # 6) PPO-style multi-epoch optimization
        for epoch in range(cfg.num_epochs_per_batch):
            # Forward pass to get current log-probs
            logits = model(input_ids=full_ids, attention_mask=full_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            token_logps = log_probs.gather(2, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            curr_comp_logps = token_logps[:, prompt_len - 1 :][:, :min_len]

            # Per-sequence current log-probs
            curr_seq_logps = (curr_comp_logps * comp_mask).sum(dim=1)

            # Importance ratio
            ratio = torch.exp(curr_seq_logps - old_seq_logps)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty against reference model
            ref_logps = get_per_token_logps(ref_model, full_ids, full_mask)
            ref_comp_logps = ref_logps[:, prompt_len - 1 :][:, :min_len]
            kl = compute_kl_divergence(curr_comp_logps, ref_comp_logps)
            kl_per_seq = (kl * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1)
            kl_loss = cfg.kl_coef * kl_per_seq.mean()

            loss = policy_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Logging
        if step % cfg.log_interval == 0:
            log(
                {
                    "reward": mean_reward,
                    "policy_loss": policy_loss.item(),
                    "kl": kl_per_seq.mean().item(),
                    "loss": loss.item(),
                    "reward_std": rewards_grouped.std(dim=1).mean().item(),
                },
                step,
            )

        # Print sample completions
        if step % cfg.sample_interval == 0:
            print(f"\n--- Sample (step {step}) ---")
            # Show first prompt with its best and worst completion
            group_rewards = rewards_t[: cfg.group_size]
            best_idx = group_rewards.argmax().item()
            print(f"Prompt: {prompts[0].strip()}")
            print(f"Best completion (reward={all_rewards[best_idx]}):")
            print(f"  {gen['completions'][best_idx].strip()}")
            print(f"Expected: {answers[0]}")
            print(f"Group rewards: {[all_rewards[j] for j in range(cfg.group_size)]}")
            print("---\n")

    print("Training complete.")


if __name__ == "__main__":
    train(Config())
