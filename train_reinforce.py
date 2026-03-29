"""REINFORCE with Verifiable Rewards.

Usage: python train_reinforce.py
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
    batch_size: int = 8
    max_steps: int = 200
    max_new_tokens: int = 256
    temperature: float = 0.7
    kl_coef: float = 0.05
    seed: int = 42
    use_wandb: bool = False
    log_interval: int = 1
    sample_interval: int = 10


def train(cfg: Config):
    set_seed(cfg.seed)
    if not torch.cuda.is_available() and cfg.device == "cuda":
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    log = setup_logging(use_wandb=cfg.use_wandb, run_name="reinforce")
    print(f"Loading model {cfg.model_name}...")
    model, tokenizer = load_model(cfg.model_name, cfg.device)
    ref_model = make_ref_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    generate_fn = get_task(cfg.task)

    reward_baseline = 0.0

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
        )

        rewards = score_completions(gen["completions"], prompts, answers, cfg.task)
        rewards_t = torch.tensor(rewards, device=cfg.device)
        mean_reward = rewards_t.mean().item()

        reward_baseline = 0.9 * reward_baseline + 0.1 * mean_reward
        advantages = rewards_t - reward_baseline

        full_ids, full_mask = gen["full_ids"], gen["full_mask"]
        prompt_len = gen["prompt_ids"].shape[1]
        comp_mask = gen["completion_mask"]
        comp_len = comp_mask.shape[1]

        comp_logps = forward_logps(model, full_ids, full_mask, prompt_len, comp_len)
        comp_mask = comp_mask[:, : comp_logps.shape[1]]
        comp_len = comp_mask.shape[1]

        seq_logps = (comp_logps * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1)
        policy_loss = -(advantages * seq_logps).mean()
        kl = kl_penalty(comp_logps, ref_model, full_ids, full_mask, prompt_len, comp_len, comp_mask)
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
            print(f"\n--- Sample (step {step}) ---")
            print(f"Prompt: {prompts[0].strip()}")
            print(f"Completion: {gen['completions'][0].strip()}")
            print(f"Expected: {answers[0]} | Reward: {rewards[0]}")
            print("---\n")

    print("Training complete.")


if __name__ == "__main__":
    train(Config())
