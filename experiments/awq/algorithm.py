import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


def pseudo_quantize_weight(weight: torch.Tensor, bits: int, group_size: int, sym: bool = False) -> torch.Tensor:
    rows, cols = weight.shape
    quantized = torch.zeros_like(weight)
    group_size = cols if group_size <= 0 else group_size
    maxq = 2**bits - 1

    for start in range(0, cols, group_size):
        end = min(start + group_size, cols)
        group = weight[:, start:end]
        if sym:
            max_val = group.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            scale = max_val / max((2 ** (bits - 1)) - 1, 1)
            q = torch.clamp(torch.round(group / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            quantized[:, start:end] = q * scale
        else:
            min_val = group.amin(dim=1, keepdim=True)
            max_val = group.amax(dim=1, keepdim=True)
            scale = ((max_val - min_val) / maxq).clamp_min(1e-8)
            zero = torch.round(-min_val / scale)
            q = torch.clamp(torch.round(group / scale) + zero, 0, maxq)
            quantized[:, start:end] = scale * (q - zero)
    return quantized


class AWQQuantizer:
    """Activation-aware weight-only quantization with per-channel scale search."""

    def __init__(self, layer, bits: int, group_size: int, alpha_grid: int = 20, sym: bool = False, max_calibration_rows: int = 256):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.bits = bits
        self.group_size = group_size
        self.alpha_grid = alpha_grid
        self.sym = sym
        self.max_calibration_rows = max_calibration_rows
        self.activation_abs_sum = None
        self.activation_count = 0
        self.sample_inputs: List[torch.Tensor] = []

    def _prepare_weight(self) -> torch.Tensor:
        weight = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        return weight.float()

    def add_batch(self, inp, out):
        del out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
        else:
            inp = inp.reshape((inp.shape[0], -1))

        inp = inp.float()
        batch_abs_sum = inp.abs().sum(dim=0)
        if self.activation_abs_sum is None:
            self.activation_abs_sum = batch_abs_sum
        else:
            self.activation_abs_sum += batch_abs_sum
        self.activation_count += int(inp.shape[0])

        if sum(item.shape[0] for item in self.sample_inputs) < self.max_calibration_rows:
            remain = self.max_calibration_rows - sum(item.shape[0] for item in self.sample_inputs)
            self.sample_inputs.append(inp[:remain].detach().cpu())

    def _build_scale(self, activation_mean: torch.Tensor, alpha: float) -> torch.Tensor:
        scale = activation_mean.clamp_min(1e-6).pow(alpha)
        scale = scale / torch.sqrt(scale.max() * scale.min()).clamp_min(1e-6)
        return scale

    def fasterquant(self) -> Dict:
        weight = self._prepare_weight()
        if self.activation_abs_sum is None or self.activation_count == 0:
            raise RuntimeError("AWQQuantizer requires calibration batches before quantization.")

        activation_mean = (self.activation_abs_sum / self.activation_count).to(self.dev)
        calibration_inputs = torch.cat(self.sample_inputs, dim=0).to(self.dev) if self.sample_inputs else None
        reference_output = calibration_inputs.matmul(weight.t()) if calibration_inputs is not None else None

        start = time.time()
        best_alpha = 0.0
        best_loss = float("inf")
        best_weight = weight

        grid_size = max(1, int(self.alpha_grid))
        for step_idx in range(grid_size + 1):
            alpha = step_idx / grid_size
            scale = self._build_scale(activation_mean, alpha)
            scaled_weight = weight * scale.unsqueeze(0)
            quantized = pseudo_quantize_weight(
                scaled_weight,
                bits=self.bits,
                group_size=self.group_size,
                sym=self.sym,
            )
            restored = quantized / scale.unsqueeze(0)

            if reference_output is not None:
                quantized_output = calibration_inputs.matmul(restored.t())
                loss = float(torch.mean((reference_output - quantized_output) ** 2).item())
            else:
                loss = float(torch.mean((weight - restored) ** 2).item())

            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
                best_weight = restored

        if isinstance(self.layer, transformers.Conv1D):
            best_weight = best_weight.t()

        self.layer.weight.data = best_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        quantized_params = int(self.layer.weight.data.numel())
        return {
            "time_s": round(time.time() - start, 4),
            "error": round(best_loss, 6),
            "bits": int(self.bits),
            "groupsize": int(self.group_size),
            "best_alpha": round(best_alpha, 4),
            "quantized_params": quantized_params,
        }

    def free(self):
        self.activation_abs_sum = None
        self.sample_inputs = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def find_linear_layers(module, prefix=""):
    if isinstance(module, nn.Linear):
        return {prefix: module}
    found = {}
    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        found.update(find_linear_layers(child, child_prefix))
    return found


def render_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{system_prompt}\n{user_prompt}"


def build_calibration_batch(tokenizer, prompts: List[str], system_prompt: str, max_length: int) -> Dict:
    rendered = [render_chat_prompt(tokenizer, system_prompt, prompt) for prompt in prompts]
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
    layers[0] = layers[0].to(device)

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
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        try:
            model(
                input_ids=input_ids[i : i + 1].to(device),
                attention_mask=attention_mask[i : i + 1].to(device) if attention_mask is not None else None,
                use_cache=False,
            )
        except ValueError:
            pass

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


def _module_selected(layer_idx: int, local_name: str, global_name: str, selected_targets: Set[str], awq_config: Dict) -> bool:
    if global_name not in selected_targets:
        return False
    min_layer = awq_config.get("min_layer", -1)
    max_layer = awq_config.get("max_layer", 10**9)
    quant_only = awq_config.get("quantize_only", "")
    invert = bool(awq_config.get("invert", False))
    base = min_layer <= layer_idx < max_layer and quant_only in local_name
    return (not base) if invert else base


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
def quantize_openpangu_awq_sequential(model, input_ids, attention_mask, device, selected_targets: Set[str], awq_config: Dict) -> Dict:
    start = time.time()
    _reset_cuda_peak_memory_stats(device)
    decoder = model.model
    layers = decoder.layers

    hidden_states, sample_kwargs = capture_decoder_inputs(model, input_ids, attention_mask, device)
    next_hidden_states = torch.zeros_like(hidden_states)

    quantized_modules = []
    layer_stats = defaultdict(lambda: {"module_count": 0, "quantized_params": 0, "total_params": 0})

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
                awq_config,
            )
        }
        execution_groups = build_execution_groups(list(all_linear.keys()), selected_local_names, bool(awq_config.get("true_sequential", True)))

        for group in execution_groups:
            subset = {name: all_linear[name] for name in group}
            quantizers = {
                name: AWQQuantizer(
                    module,
                    bits=int(awq_config["bits"]),
                    group_size=int(awq_config.get("group_size", 128)),
                    alpha_grid=int(awq_config.get("alpha_grid", 20)),
                    sym=bool(awq_config.get("sym", False)),
                    max_calibration_rows=int(awq_config.get("max_calibration_rows", 256)),
                )
                for name, module in subset.items()
            }

            def add_batch(name):
                def hook(_, inp, out):
                    quantizers[name].add_batch(inp[0].detach(), out.detach())

                return hook

            handles = [module.register_forward_hook(add_batch(name)) for name, module in subset.items()]
            for sample_idx in range(hidden_states.shape[0]):
                next_hidden_states[sample_idx] = run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device)
            for handle in handles:
                handle.remove()

            for name in group:
                quant_metrics = quantizers[name].fasterquant()
                total_params = int(subset[name].weight.numel())
                global_name = f"model.layers.{layer_idx}.{name}"
                quantized_modules.append(
                    {
                        "layer": int(layer_idx),
                        "name": global_name,
                        "total_params": total_params,
                        **quant_metrics,
                    }
                )
                layer_stats[layer_idx]["module_count"] += 1
                layer_stats[layer_idx]["quantized_params"] += quant_metrics["quantized_params"]
                layer_stats[layer_idx]["total_params"] += total_params
                quantizers[name].free()

        for sample_idx in range(hidden_states.shape[0]):
            next_hidden_states[sample_idx] = run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device)

        layers[layer_idx] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hidden_states, next_hidden_states = next_hidden_states, hidden_states

    total_quantized_params = sum(item["quantized_params"] for item in quantized_modules)
    total_params = sum(item["total_params"] for item in quantized_modules)

    return {
        "module_count": len(quantized_modules),
        "total_quantized_params": int(total_quantized_params),
        "total_target_params": int(total_params),
        "quantized_fraction": round(total_quantized_params / total_params, 6) if total_params else 0.0,
        "elapsed_s": round(time.time() - start, 4),
        "peak_memory_mb": _get_cuda_peak_memory_mb(device),
        "modules": quantized_modules,
        "layers": [
            {
                "layer": int(layer_idx),
                "module_count": int(stats["module_count"]),
                "quantized_params": int(stats["quantized_params"]),
                "total_params": int(stats["total_params"]),
                "quantized_fraction": round(stats["quantized_params"] / stats["total_params"], 6) if stats["total_params"] else 0.0,
            }
            for layer_idx, stats in sorted(layer_stats.items())
        ],
    }
