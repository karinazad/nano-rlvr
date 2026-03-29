"""Shared utilities for RLVR training."""

import random

import numpy as np
import torch


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
