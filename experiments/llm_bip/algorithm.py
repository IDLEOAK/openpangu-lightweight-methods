import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class _CaptureInputsExit(Exception):
    """Internal sentinel used to stop the forward pass after layer-0 inputs are captured."""


def find_linear_layers(module, prefix=""):
    if isinstance(module, nn.Linear):
        return {prefix: module}
    found = {}
    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        found.update(find_linear_layers(child, child_prefix))
    return found


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


def build_execution_groups(all_names: List[str], selected_names: Set[str], true_sequential: bool) -> List[List[str]]:
    if not true_sequential:
        ordered = [name for name in all_names if name in selected_names]
        return [ordered] if ordered else []

    canonical = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    groups = []
    seen = set()
    for group in canonical:
        current = [name for name in group if name in selected_names]
        if current:
            groups.append(current)
            seen.update(current)
    remaining = [name for name in all_names if name in selected_names and name not in seen]
    if remaining:
        groups.append(remaining)
    return groups


def _module_selected(layer_idx: int, local_name: str, global_name: str, selected_targets: Set[str], bip_config: Dict) -> bool:
    if global_name not in selected_targets:
        return False
    min_layer = bip_config.get("min_layer", -1)
    max_layer = bip_config.get("max_layer", 10**9)
    prune_only = bip_config.get("prune_only", "")
    invert = bool(bip_config.get("invert", False))
    base = min_layer <= layer_idx < max_layer and prune_only in local_name
    return (not base) if invert else base


def _zero_output_group(module, start: int, end: int):
    module.weight.data[start:end].zero_()
    if getattr(module, "bias", None) is not None:
        module.bias.data[start:end].zero_()


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


@torch.no_grad()
def prune_openpangu_llm_bip_sequential(model, input_ids, attention_mask, device, selected_targets: Set[str], bip_config: Dict) -> Dict:
    start = time.time()
    _reset_cuda_peak_memory_stats(device)
    decoder = model.model
    layers = decoder.layers

    hidden_states, sample_kwargs = capture_decoder_inputs(model, input_ids, attention_mask, device)
    next_hidden_states = torch.zeros_like(hidden_states)

    pruned_modules = []
    layer_stats = defaultdict(lambda: {"module_count": 0, "zero_params": 0, "total_params": 0})
    scoring_samples = max(1, int(bip_config.get("scoring_samples", 1)))
    group_size = max(1, int(bip_config.get("group_size", 1024)))

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(device)
        all_linear = find_linear_layers(layer)
        selected_local_names = {
            local_name
            for local_name in all_linear
            if _module_selected(
                layer_idx,
                local_name,
                f"model.layers.{layer_idx}.{local_name}",
                selected_targets,
                bip_config,
            )
        }
        execution_groups = build_execution_groups(list(all_linear.keys()), selected_local_names, bool(bip_config.get("true_sequential", True)))

        baseline_outputs = []
        baseline_count = min(scoring_samples, hidden_states.shape[0])
        for sample_idx in range(baseline_count):
            baseline_outputs.append(run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device))

        for group in execution_groups:
            subset = {name: all_linear[name] for name in group}
            candidates = []
            module_meta = {}

            for name, module in subset.items():
                weight = module.weight.data
                out_features = weight.shape[0]
                groups = [(start_idx, min(start_idx + group_size, out_features)) for start_idx in range(0, out_features, group_size)]
                if len(groups) <= 1:
                    continue

                original_weight = weight.clone()
                original_bias = module.bias.data.clone() if module.bias is not None else None
                module_meta[name] = {
                    "module": module,
                    "original_weight": original_weight,
                    "original_bias": original_bias,
                    "groups": groups,
                    "total_params": int(module.weight.data.numel()),
                }

                for group_index, (start_idx, end_idx) in enumerate(groups):
                    _zero_output_group(module, start_idx, end_idx)
                    loss = 0.0
                    for sample_idx in range(baseline_count):
                        baseline = baseline_outputs[sample_idx]
                        perturbed = run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device)
                        mse = torch.mean((baseline - perturbed) ** 2)
                        denom = torch.mean(baseline.float() ** 2).clamp_min(1e-8)
                        loss += float((mse / denom).item())
                    candidates.append(
                        {
                            "name": name,
                            "group_index": group_index,
                            "start": start_idx,
                            "end": end_idx,
                            "score": loss / baseline_count,
                            "param_count": int((end_idx - start_idx) * module.weight.data.shape[1]),
                        }
                    )
                    module.weight.data.copy_(original_weight)
                    if module.bias is not None:
                        module.bias.data.copy_(original_bias)

            if not candidates:
                continue

            total_candidate_params = sum(item["param_count"] for item in candidates)
            target_prune_params = max(1, int(total_candidate_params * float(bip_config["sparsity"])))
            selected_candidates = []
            accumulated = 0
            for item in sorted(candidates, key=lambda x: x["score"]):
                if accumulated >= target_prune_params:
                    break
                selected_candidates.append(item)
                accumulated += item["param_count"]

            selected_by_module = defaultdict(list)
            for item in selected_candidates:
                selected_by_module[item["name"]].append(item)

            for name, meta in module_meta.items():
                module = meta["module"]
                selected_items = selected_by_module.get(name, [])
                for item in selected_items:
                    _zero_output_group(module, item["start"], item["end"])

                zero_params = int((module.weight.data == 0).sum().item())
                total_params = int(module.weight.data.numel())
                global_name = f"model.layers.{layer_idx}.{name}"
                pruned_modules.append(
                    {
                        "layer": int(layer_idx),
                        "name": global_name,
                        "group_size": group_size,
                        "group_count": len(meta["groups"]),
                        "pruned_groups": len(selected_items),
                        "zero_params": zero_params,
                        "total_params": total_params,
                        "zero_fraction": round(zero_params / total_params, 6) if total_params else 0.0,
                    }
                )
                layer_stats[layer_idx]["module_count"] += 1
                layer_stats[layer_idx]["zero_params"] += zero_params
                layer_stats[layer_idx]["total_params"] += total_params

        for sample_idx in range(hidden_states.shape[0]):
            next_hidden_states[sample_idx] = run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device)

        layers[layer_idx] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hidden_states, next_hidden_states = next_hidden_states, hidden_states

    total_zero_params = sum(item["zero_params"] for item in pruned_modules)
    total_params = sum(item["total_params"] for item in pruned_modules)

    return {
        "module_count": len(pruned_modules),
        "total_zero_params": int(total_zero_params),
        "total_pruned_params": int(total_params),
        "overall_zero_fraction": round(total_zero_params / total_params, 6) if total_params else 0.0,
        "elapsed_s": round(time.time() - start, 4),
        "peak_memory_mb": _get_cuda_peak_memory_mb(device),
        "modules": pruned_modules,
        "layers": [
            {
                "layer": int(layer_idx),
                "module_count": int(stats["module_count"]),
                "zero_params": int(stats["zero_params"]),
                "total_params": int(stats["total_params"]),
                "zero_fraction": round(stats["zero_params"] / stats["total_params"], 6) if stats["total_params"] else 0.0,
            }
            for layer_idx, stats in sorted(layer_stats.items())
        ],
    }
