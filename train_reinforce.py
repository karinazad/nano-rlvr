"""REINFORCE with Verifiable Rewards — the simplest RLVR algorithm.

This is the "hello world" of RLVR: generate completions, check if the answer
is correct, and use the reward signal to update the policy via REINFORCE.

Usage:
    python train_reinforce.py
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
    batch_size: int = 8
    max_steps: int = 200
    max_new_tokens: int = 256
    temperature: float = 0.7
    kl_coef: float = 0.05  # KL penalty coefficient
    seed: int = 42
    use_wandb: bool = False
    log_interval: int = 1
    sample_interval: int = 10  # print sample completions every N steps


def train(cfg: Config):
    set_seed(cfg.seed)

    if not torch.cuda.is_available() and cfg.device == "cuda":
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    log = setup_logging(use_wandb=cfg.use_wandb, run_name="reinforce")

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

    reward_baseline = 0.0  # running mean for variance reduction

    for step in range(1, cfg.max_steps + 1):
        model.train()

        # 1) Sample problems
        problems = generate_fn(cfg.batch_size)
        prompts = [p for p, _ in problems]
        answers = [a for _, a in problems]

        # 2) Generate completions
        gen = generate_completions(
            model,
            tokenizer,
            prompts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

        # 3) Compute rewards
        rewards = []
        for i, completion in enumerate(gen["completions"]):
            if cfg.task == "arithmetic":
                r = check_arithmetic(completion, answers[i])
            else:
                # For countdown, we need the numbers from the prompt
                nums_str = prompts[i].split("[")[1].split("]")[0]
                numbers = [int(n.strip()) for n in nums_str.split(",")]
                r = check_countdown(completion, numbers, answers[i])
            rewards.append(r)

        rewards_t = torch.tensor(rewards, device=cfg.device)
        mean_reward = rewards_t.mean().item()

        # 4) Baseline subtraction (running mean)
        reward_baseline = 0.9 * reward_baseline + 0.1 * mean_reward
        advantages = rewards_t - reward_baseline

        # 5) Compute log-probs under current policy (with gradients)
        full_ids = gen["full_ids"]
        full_mask = gen["full_mask"]
        prompt_len = gen["prompt_ids"].shape[1]

        logits = model(input_ids=full_ids, attention_mask=full_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        token_logps = log_probs.gather(2, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        comp_logps = token_logps[:, prompt_len - 1 :]

        comp_mask = gen["completion_mask"]
        min_len = min(comp_logps.shape[1], comp_mask.shape[1])
        comp_logps = comp_logps[:, :min_len]
        comp_mask = comp_mask[:, :min_len]

        # 6) KL penalty against reference model
        ref_logps = get_per_token_logps(ref_model, full_ids, full_mask)
        ref_comp_logps = ref_logps[:, prompt_len - 1 :][:, :min_len]
        kl = compute_kl_divergence(comp_logps, ref_comp_logps)
        kl_per_seq = (kl * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1)

        # 7) REINFORCE loss: -advantage * sum(log_probs) + kl_penalty
        seq_logps = (comp_logps * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1)
        policy_loss = -(advantages * seq_logps).mean()
        kl_loss = cfg.kl_coef * kl_per_seq.mean()
        loss = policy_loss + kl_loss

        # 8) Update
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
                },
                step,
            )

        # Print sample completions
        if step % cfg.sample_interval == 0:
            print(f"\n--- Sample (step {step}) ---")
            idx = 0
            print(f"Prompt: {prompts[idx].strip()}")
            print(f"Completion: {gen['completions'][idx].strip()}")
            print(f"Expected: {answers[idx]} | Reward: {rewards[idx]}")
            print("---\n")

    print("Training complete.")


if __name__ == "__main__":
    train(Config())
