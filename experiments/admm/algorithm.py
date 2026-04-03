import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class ADMMPruner:
    """Layer-wise ADMM weight update with Wanda-style preconditioning."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        weight = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.gram = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.sample_count = 0

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
        self.gram += inp.t().matmul(inp)
        self.sample_count += int(inp.shape[0])

    def _prepare_weight(self) -> torch.Tensor:
        weight = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        return weight.float()

    def _build_keep_mask(self, values: torch.Tensor, sparsity: float) -> torch.Tensor:
        total = values.numel()
        prune_count = int(total * sparsity)
        if prune_count <= 0:
            return torch.ones_like(values, dtype=torch.bool)
        if prune_count >= total:
            return torch.zeros_like(values, dtype=torch.bool)
        flat = values.flatten()
        _, indices = torch.topk(flat, k=prune_count, largest=False)
        keep = torch.ones_like(flat, dtype=torch.bool)
        keep[indices] = False
        return keep.view_as(values)

    def fasterprune(
        self,
        sparsity: float,
        update_steps: int = 20,
        sparsify_steps: int = 15,
        rho: float = 1.0,
        ridge: float = 0.1,
        gradual: bool = True,
    ) -> Dict:
        weight = self._prepare_weight()
        feature_norm = torch.sqrt(torch.diag(self.gram)).clamp_min(1e-8)

        # Wanda-style preconditioning: weight columns are scaled by input norms.
        weight_pre = weight * feature_norm.unsqueeze(0)
        inv_norm = feature_norm.reciprocal()
        xx = self.gram * inv_norm.unsqueeze(1) * inv_norm.unsqueeze(0)
        xx = xx + ridge * torch.eye(self.columns, device=self.dev, dtype=xx.dtype)

        xx_inv = torch.linalg.inv(xx + rho * torch.eye(self.columns, device=self.dev, dtype=xx.dtype))
        xx_weight = xx.matmul(weight_pre.t())

        current = weight_pre.t().clone()
        z = current.clone()
        u = torch.zeros_like(current)
        keep_mask = torch.ones_like(current, dtype=torch.bool)
        start = time.time()

        effective_steps = max(1, int(update_steps))
        effective_sparsify_steps = max(1, min(int(sparsify_steps), effective_steps))

        for step_idx in range(1, effective_steps + 1):
            if gradual:
                if step_idx <= effective_sparsify_steps:
                    step_sparsity = sparsity * (step_idx / effective_sparsify_steps) ** 3
                    keep_mask = self._build_keep_mask(torch.abs(current + u), step_sparsity)
            elif step_idx == 1:
                keep_mask = self._build_keep_mask(torch.abs(current + u), sparsity)

            z = (current + u) * keep_mask
            u = u + (current - z)
            current = xx_inv.matmul(xx_weight + rho * (z - u))

        final_weight = ((current + u) * keep_mask).t()
        final_weight = final_weight / feature_norm.unsqueeze(0)

        if isinstance(self.layer, transformers.Conv1D):
            final_weight = final_weight.t()

        self.layer.weight.data = final_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        zero_params = int((self.layer.weight.data == 0).sum().item())
        total_params = int(self.layer.weight.data.numel())
        error = float(torch.mean((weight - final_weight) ** 2).item())

        return {
            "time_s": round(time.time() - start, 4),
            "error": round(error, 6),
            "zero_params": zero_params,
            "total_params": total_params,
            "zero_fraction": round(zero_params / total_params, 6) if total_params else 0.0,
        }

    def free(self):
        self.gram = None
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


def _module_selected(layer_idx: int, local_name: str, global_name: str, selected_targets: Set[str], admm_config: Dict) -> bool:
    if global_name not in selected_targets:
        return False
    min_layer = admm_config.get("min_layer", -1)
    max_layer = admm_config.get("max_layer", 10**9)
    prune_only = admm_config.get("prune_only", "")
    invert = bool(admm_config.get("invert", False))
    base = min_layer <= layer_idx < max_layer and prune_only in local_name
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
def prune_openpangu_admm_sequential(model, input_ids, attention_mask, device, selected_targets: Set[str], admm_config: Dict) -> Dict:
    start = time.time()
    _reset_cuda_peak_memory_stats(device)
    decoder = model.model
    layers = decoder.layers

    hidden_states, sample_kwargs = capture_decoder_inputs(model, input_ids, attention_mask, device)
    next_hidden_states = torch.zeros_like(hidden_states)

    pruned_modules = []
    layer_stats = defaultdict(lambda: {"module_count": 0, "zero_params": 0, "total_params": 0})

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
                admm_config,
            )
        }
        execution_groups = build_execution_groups(list(all_linear.keys()), selected_local_names, bool(admm_config.get("true_sequential", True)))

        for group in execution_groups:
            subset = {name: all_linear[name] for name in group}
            pruners = {name: ADMMPruner(module) for name, module in subset.items()}

            def add_batch(name):
                def hook(_, inp, out):
                    pruners[name].add_batch(inp[0].detach(), out.detach())

                return hook

            handles = [module.register_forward_hook(add_batch(name)) for name, module in subset.items()]
            for sample_idx in range(hidden_states.shape[0]):
                next_hidden_states[sample_idx] = run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device)
            for handle in handles:
                handle.remove()

            for name in group:
                prune_metrics = pruners[name].fasterprune(
                    sparsity=float(admm_config["sparsity"]),
                    update_steps=int(admm_config.get("update_steps", 20)),
                    sparsify_steps=int(admm_config.get("sparsify_steps", 15)),
                    rho=float(admm_config.get("rho", 1.0)),
                    ridge=float(admm_config.get("ridge", 0.1)),
                    gradual=bool(admm_config.get("gradual", True)),
                )
                global_name = f"model.layers.{layer_idx}.{name}"
                pruned_modules.append(
                    {
                        "layer": int(layer_idx),
                        "name": global_name,
                        **prune_metrics,
                    }
                )
                layer_stats[layer_idx]["module_count"] += 1
                layer_stats[layer_idx]["zero_params"] += prune_metrics["zero_params"]
                layer_stats[layer_idx]["total_params"] += prune_metrics["total_params"]
                pruners[name].free()

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
