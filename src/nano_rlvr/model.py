"""Model loading and generation utilities for RLVR."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="Qwen/Qwen2.5-0.5B", device="cuda"):
    """Load a pretrained causal LM and tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        Device to load the model onto.

    Returns
    -------
    model : AutoModelForCausalLM
        The loaded model.
    tokenizer : AutoTokenizer
        The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    return model, tokenizer


def get_per_token_logps(model, input_ids, attention_mask):
    """Compute per-token log-probabilities via a forward pass.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The language model.
    input_ids : torch.Tensor
        Token ids, shape (batch, seq_len).
    attention_mask : torch.Tensor
        Attention mask, shape (batch, seq_len).

    Returns
    -------
    torch.Tensor
        Per-token log-probs for tokens at positions 1..seq_len-1,
        shape (batch, seq_len - 1).
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # logits[:, t, :] predicts token at position t+1
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    # Gather the log-probs of the actual next tokens
    token_logps = log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_logps


def generate_completions(
    model, tokenizer, prompts, max_new_tokens=256, temperature=0.7, num_samples=1
):
    """Generate completions and collect per-token log-probs.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The language model.
    tokenizer : AutoTokenizer
        The tokenizer.
    prompts : list of str
        Input prompts.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    num_samples : int
        Number of completions per prompt.

    Returns
    -------
    dict
        Keys:
        - "completions": list of str, length len(prompts) * num_samples
        - "prompt_ids": padded prompt token ids (batch, prompt_len)
        - "completion_ids": padded completion token ids (batch, comp_len)
        - "full_ids": padded full sequence token ids (batch, full_len)
        - "full_mask": attention mask for full_ids
        - "completion_mask": mask over just the completion tokens (batch, comp_len)
        - "logps": per-token log-probs over completion tokens (batch, comp_len)
    """
    device = next(model.parameters()).device

    # Duplicate prompts for num_samples
    expanded_prompts = [p for p in prompts for _ in range(num_samples)]

    # Tokenize prompts with left-padding for batch generation
    tokenizer.padding_side = "left"
    prompt_enc = tokenizer(expanded_prompts, return_tensors="pt", padding=True)
    prompt_ids = prompt_enc["input_ids"].to(device)
    prompt_mask = prompt_enc["attention_mask"].to(device)
    prompt_len = prompt_ids.shape[1]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Split into prompt and completion parts
    completion_ids = output_ids[:, prompt_len:]
    full_mask = (output_ids != tokenizer.pad_token_id).long()
    completion_mask = full_mask[:, prompt_len:]

    # Decode completions
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    # Get per-token log-probs over the full sequence
    with torch.no_grad():
        logits = model(input_ids=output_ids, attention_mask=full_mask).logits
    log_probs = torch.log_softmax(logits[:, :-1, :] / max(temperature, 1e-8), dim=-1)
    all_logps = log_probs.gather(2, output_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    # Slice to completion portion only (positions prompt_len-1 onward in the logps tensor)
    comp_logps = all_logps[:, prompt_len - 1 :]
    # Trim to match completion_ids length
    min_len = min(comp_logps.shape[1], completion_ids.shape[1])
    comp_logps = comp_logps[:, :min_len]
    completion_mask = completion_mask[:, :min_len]
    comp_logps = comp_logps * completion_mask  # zero out padding

    return {
        "completions": completions,
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids[:, :min_len],
        "full_ids": output_ids,
        "full_mask": full_mask,
        "completion_mask": completion_mask,
        "logps": comp_logps,
    }
