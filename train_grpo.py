"""Group Relative Policy Optimization (GRPO) with Verifiable Rewards.

Usage: python train_grpo.py
"""

from dataclasses import dataclass

import torch

from nano_rlvr.model import generate_completions, load_model
from nano_rlvr.utils import (
    forward_logps,
    get_task,
    kl_penalty,
    make_ref_model,
    score_completions,
    set_seed,
    setup_logging,
)


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda"
    task: str = "arithmetic"
    lr: float = 1e-6
    batch_size: int = 4
    group_size: int = 8
    max_steps: int = 200
    max_new_tokens: int = 256
    temperature: float = 0.7
    clip_epsilon: float = 0.2
    kl_coef: float = 0.05
    num_epochs_per_batch: int = 2
    seed: int = 42
    use_wandb: bool = False
    log_interval: int = 1
    sample_interval: int = 10


def compute_group_advantages(rewards_t, batch_size, group_size, device):
    """Normalize rewards within each group (no critic needed)."""
    grouped = rewards_t.view(batch_size, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    adv = (grouped - mean) / (std + 1e-8)
    adv[std.squeeze(-1) < 1e-8] = 0.0
    return adv.view(-1)


def train(cfg: Config):
    set_seed(cfg.seed)
    if not torch.cuda.is_available() and cfg.device == "cuda":
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    log = setup_logging(use_wandb=cfg.use_wandb, run_name="grpo")
    print(f"Loading model {cfg.model_name}...")
    model, tokenizer = load_model(cfg.model_name, cfg.device)
    ref_model = make_ref_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    generate_fn = get_task(cfg.task)

    for step in range(1, cfg.max_steps + 1):
        model.train()

        problems = generate_fn(cfg.batch_size)
        prompts = [p for p, _ in problems]
        answers = [a for _, a in problems]

        gen = generate_completions(
            model,
            tokenizer,
            prompts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            num_samples=cfg.group_size,
        )

        rewards = score_completions(
            gen["completions"],
            prompts,
            answers,
            cfg.task,
            group_size=cfg.group_size,
        )
        rewards_t = torch.tensor(rewards, device=cfg.device)
        mean_reward = rewards_t.mean().item()
        advantages = compute_group_advantages(rewards_t, cfg.batch_size, cfg.group_size, cfg.device)

        # Old log-probs from generation (frozen)
        old_logps = gen["logps"].detach()
        full_ids, full_mask = gen["full_ids"], gen["full_mask"]
        prompt_len = gen["prompt_ids"].shape[1]
        comp_mask = gen["completion_mask"]

        comp_len = min(old_logps.shape[1], comp_mask.shape[1])
        old_logps = old_logps[:, :comp_len]
        comp_mask = comp_mask[:, :comp_len]
        old_seq_logps = (old_logps * comp_mask).sum(dim=1)

        # PPO-style multi-epoch optimization
        for _ in range(cfg.num_epochs_per_batch):
            curr_logps = forward_logps(model, full_ids, full_mask, prompt_len, comp_len)
            curr_seq_logps = (curr_logps * comp_mask).sum(dim=1)

            ratio = torch.exp(curr_seq_logps - old_seq_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            kl = kl_penalty(
                curr_logps,
                ref_model,
                full_ids,
                full_mask,
                prompt_len,
                comp_len,
                comp_mask,
            )
            loss = policy_loss + cfg.kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % cfg.log_interval == 0:
            log(
                {
                    "reward": mean_reward,
                    "policy_loss": policy_loss.item(),
                    "kl": kl.item(),
                    "loss": loss.item(),
                },
                step,
            )

        if step % cfg.sample_interval == 0:
            group_rewards = rewards_t[: cfg.group_size]
            best_idx = group_rewards.argmax().item()
            print(f"\n--- Sample (step {step}) ---")
            print(f"Prompt: {prompts[0].strip()}")
            print(f"Best completion (reward={rewards[best_idx]}):")
            print(f"  {gen['completions'][best_idx].strip()}")
            print(f"Expected: {answers[0]}")
            print(f"Group rewards: {rewards[: cfg.group_size]}")
            print("---\n")

    print("Training complete.")


if __name__ == "__main__":
    train(Config())
