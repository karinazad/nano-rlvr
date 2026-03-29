"""Shared utilities for RLVR training."""

import copy
import random

import numpy as np
import torch

from nano_rlvr.data import generate_arithmetic_problems, generate_countdown_problems
from nano_rlvr.model import get_per_token_logps
from nano_rlvr.rewards import check_arithmetic, check_countdown


def get_task(task_name):
    """Return the problem generator for a given task name.

    Parameters
    ----------
    task_name : str
        One of "arithmetic" or "countdown".

    Returns
    -------
    callable
        Problem generator function.
    """
    tasks = {
        "arithmetic": generate_arithmetic_problems,
        "countdown": generate_countdown_problems,
    }
    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(tasks)}")
    return tasks[task_name]


def score_completions(completions, prompts, answers, task_name, group_size=1):
    """Compute verifiable rewards for a batch of completions.

    Parameters
    ----------
    completions : list of str
        Model completions.
    prompts : list of str
        Original prompts (one per group of completions).
    answers : list
        Expected answers (one per prompt).
    task_name : str
        Task type for reward dispatch.
    group_size : int
        Number of completions per prompt.

    Returns
    -------
    list of float
        Reward for each completion.
    """
    rewards = []
    for i, completion in enumerate(completions):
        prompt_idx = i // group_size
        if task_name == "arithmetic":
            r = check_arithmetic(completion, answers[prompt_idx])
        else:
            nums_str = prompts[prompt_idx].split("[")[1].split("]")[0]
            numbers = [int(n.strip()) for n in nums_str.split(",")]
            r = check_countdown(completion, numbers, answers[prompt_idx])
        rewards.append(r)
    return rewards


def make_ref_model(model):
    """Create a frozen copy of a model for KL reference.

    Parameters
    ----------
    model : torch.nn.Module
        The model to copy.

    Returns
    -------
    torch.nn.Module
        Frozen deepcopy.
    """
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def forward_logps(model, full_ids, full_mask, prompt_len, comp_len):
    """Forward pass to get per-token log-probs over the completion portion.

    Parameters
    ----------
    model : torch.nn.Module
        The language model.
    full_ids : torch.Tensor
        Full token ids (prompt + completion), shape (batch, seq_len).
    full_mask : torch.Tensor
        Attention mask, shape (batch, seq_len).
    prompt_len : int
        Length of the prompt portion.
    comp_len : int
        Length to truncate the completion portion to.

    Returns
    -------
    torch.Tensor
        Per-token log-probs over completion, shape (batch, comp_len).
    """
    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    token_logps = log_probs.gather(2, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_logps[:, prompt_len - 1 :][:, :comp_len]


def kl_penalty(comp_logps, ref_model, full_ids, full_mask, prompt_len, comp_len, comp_mask):
    """Compute mean per-sequence KL divergence against a reference model.

    Parameters
    ----------
    comp_logps : torch.Tensor
        Current policy completion log-probs, shape (batch, comp_len).
    ref_model : torch.nn.Module
        Frozen reference model.
    full_ids : torch.Tensor
        Full token ids.
    full_mask : torch.Tensor
        Attention mask.
    prompt_len : int
        Prompt length.
    comp_len : int
        Completion length.
    comp_mask : torch.Tensor
        Completion mask.

    Returns
    -------
    torch.Tensor
        Scalar mean KL divergence.
    """
    ref_logps = get_per_token_logps(ref_model, full_ids, full_mask)
    ref_comp_logps = ref_logps[:, prompt_len - 1 :][:, :comp_len]
    kl = comp_logps - ref_comp_logps
    kl_per_seq = (kl * comp_mask).sum(dim=1) / comp_mask.sum(dim=1).clamp(min=1)
    return kl_per_seq.mean()


def compute_kl_divergence(logps_current, logps_reference):
    """Compute per-token KL divergence: KL(current || reference).

    Parameters
    ----------
    logps_current : torch.Tensor
        Log-probs under the current policy, shape (batch, seq_len).
    logps_reference : torch.Tensor
        Log-probs under the reference policy, shape (batch, seq_len).

    Returns
    -------
    torch.Tensor
        Per-token KL divergence, shape (batch, seq_len).
    """
    return logps_current - logps_reference


def normalize_advantages(advantages):
    """Zero-mean, unit-variance normalization.

    Parameters
    ----------
    advantages : torch.Tensor
        Raw advantage values.

    Returns
    -------
    torch.Tensor
        Normalized advantages (returns zeros if std is too small).
    """
    std = advantages.std()
    if std < 1e-8:
        return torch.zeros_like(advantages)
    return (advantages - advantages.mean()) / (std + 1e-8)


def setup_logging(use_wandb=False, project_name="nano-rlvr", run_name=None):
    """Set up logging (print + optional wandb).

    Parameters
    ----------
    use_wandb : bool
        Whether to initialize wandb.
    project_name : str
        Wandb project name.
    run_name : str, optional
        Wandb run name.

    Returns
    -------
    callable
        A log function that takes a dict of metrics and a step number.
    """
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(project=project_name, name=run_name)

    def log_fn(metrics, step):
        parts = [f"step {step}"]
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print(" | ".join(parts))
        if wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    return log_fn


def set_seed(seed):
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
