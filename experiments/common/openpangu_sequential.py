import math
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CaptureInputsExit(Exception):
    """Internal sentinel used to stop the forward pass after layer-0 inputs are captured."""


def render_chat_prompt(tokenizer, system_prompt: str, user_prompt: str, apply_chat_template: bool = True) -> str:
    if not apply_chat_template:
        return user_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{system_prompt}\n{user_prompt}"


def build_calibration_batch(
    tokenizer,
    prompts: List[str],
    system_prompt: str,
    max_length: int,
    apply_chat_template: bool = True,
) -> Dict:
    rendered = [render_chat_prompt(tokenizer, system_prompt, prompt, apply_chat_template) for prompt in prompts]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        "rendered_prompts": rendered,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
        "sequence_length": int(encoded["input_ids"].shape[1]),
    }


def _to_cpu(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _to_device(value, device):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _reset_cuda_peak_memory_stats(device) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def _get_cuda_peak_memory_mb(device) -> Optional[float]:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    return None


@torch.no_grad()
def capture_decoder_inputs(model, input_ids, attention_mask, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    decoder = model.model
    layers = decoder.layers
    decoder.embed_tokens = decoder.embed_tokens.to(device)
    decoder.rotary_emb = decoder.rotary_emb.to(device)
    layer0 = layers[0].to(device)
    layers[0] = layer0

    dtype = next(iter(model.parameters())).dtype
    nsamples = input_ids.shape[0]
    sequence_length = input_ids.shape[1]
    hidden_size = model.config.hidden_size
    hidden_states = torch.zeros((nsamples, sequence_length, hidden_size), dtype=dtype, device="cpu")
    sample_kwargs = []
    state = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            hidden_states[state["i"]] = inp[0].detach().cpu()
            sample_kwargs.append(
                {
                    "attention_mask": _to_cpu(kwargs.get("attention_mask")),
                    "position_ids": _to_cpu(kwargs.get("position_ids")),
                    "cache_position": _to_cpu(kwargs.get("cache_position")),
                    "position_embeddings": _to_cpu(kwargs.get("position_embeddings")),
                }
            )
            state["i"] += 1
            raise _CaptureInputsExit

    layers[0] = Catcher(layers[0])
    try:
        for i in range(nsamples):
            try:
                model(
                    input_ids=input_ids[i : i + 1].to(device),
                    attention_mask=attention_mask[i : i + 1].to(device) if attention_mask is not None else None,
                    use_cache=False,
                )
            except _CaptureInputsExit:
                pass
    finally:
        if isinstance(layers[0], Catcher):
            layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        decoder.embed_tokens = decoder.embed_tokens.cpu()
        decoder.rotary_emb = decoder.rotary_emb.cpu()
        model.config.use_cache = use_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return hidden_states, sample_kwargs


@torch.no_grad()
def run_layer(layer, hidden_state_cpu, sample_kwargs_cpu, device):
    kwargs = {
        "attention_mask": _to_device(sample_kwargs_cpu.get("attention_mask"), device),
        "position_ids": _to_device(sample_kwargs_cpu.get("position_ids"), device),
        "cache_position": _to_device(sample_kwargs_cpu.get("cache_position"), device),
        "position_embeddings": _to_device(sample_kwargs_cpu.get("position_embeddings"), device),
        "past_key_value": None,
        "output_attentions": False,
        "use_cache": False,
    }
    output = layer(hidden_state_cpu.unsqueeze(0).to(device), **kwargs)[0]
    return output.squeeze(0).detach().cpu()


@torch.no_grad()
def evaluate_openpangu_perplexity_sequential(model, input_ids, attention_mask, device):
    start = time.time()
    _reset_cuda_peak_memory_stats(device)
    decoder = model.model
    layers = decoder.layers

    hidden_states, sample_kwargs = capture_decoder_inputs(model, input_ids, attention_mask, device)
    next_hidden_states = torch.zeros_like(hidden_states)

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(device)
        for sample_idx in range(hidden_states.shape[0]):
            next_hidden_states[sample_idx] = run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device)
        layers[layer_idx] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hidden_states, next_hidden_states = next_hidden_states, hidden_states

    decoder.norm = decoder.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    token_loss_sum = 0.0
    token_count = 0
    for sample_idx in range(hidden_states.shape[0]):
        final_hidden = decoder.norm(hidden_states[sample_idx].unsqueeze(0).to(device))
        logits = model.lm_head(final_hidden).float()
        labels = input_ids[sample_idx : sample_idx + 1].to(device)
        labels_mask = attention_mask[sample_idx : sample_idx + 1].to(device) if attention_mask is not None else None

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = labels_mask[:, 1:].contiguous().bool() if labels_mask is not None else torch.ones_like(shift_labels, dtype=torch.bool)

        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        valid_losses = losses[shift_mask.view(-1)]
        token_loss_sum += float(valid_losses.sum().item())
        token_count += int(shift_mask.sum().item())

    decoder.norm = decoder.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if token_count == 0:
        return {
            "sample_count": int(input_ids.shape[0]),
            "token_count": 0,
            "avg_loss": None,
            "perplexity": None,
            "elapsed_s": round(time.time() - start, 4),
            "peak_memory_mb": _get_cuda_peak_memory_mb(device),
        }

    avg_loss = token_loss_sum / token_count
    return {
        "sample_count": int(input_ids.shape[0]),
        "token_count": int(token_count),
        "avg_loss": round(avg_loss, 6),
        "perplexity": round(math.exp(avg_loss), 6),
        "elapsed_s": round(time.time() - start, 4),
        "peak_memory_mb": _get_cuda_peak_memory_mb(device),
    }
