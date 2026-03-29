import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


def quantize_tensor(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = 1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize_tensor(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q = torch.abs(q - x).pow(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0] if weight else (shape[1] if len(shape) != 3 else shape[2])
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:
    """Minimal GPTQ port adapted from the official IST-DASLab implementation."""

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
        self.hessian = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.hessian *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.hessian += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
    ):
        weight = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()
        weight = weight.float()

        start = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(weight, weight=True)

        hessian = self.hessian
        del self.hessian
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        groups = None
        if static_groups and groupsize != -1:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = Quantizer()
                quantizer.configure(
                    int(self.quantizer.maxq.log2().item()),
                    perchannel=self.quantizer.perchannel,
                    sym=self.quantizer.sym,
                    mse=self.quantizer.mse,
                )
                quantizer.find_params(weight[:, i : (i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(hessian), descending=True)
            weight = weight[:, perm]
            hessian = hessian[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        losses = torch.zeros_like(weight)
        quantized_weight = torch.zeros_like(weight)

        damp = percdamp * torch.mean(torch.diag(hessian))
        diag = torch.arange(self.columns, device=self.dev)
        hessian[diag, diag] += damp
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian = torch.linalg.cholesky(hessian, upper=True)
        hessian_inv = hessian

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            weight_block = weight[:, i1:i2].clone()
            quant_block = torch.zeros_like(weight_block)
            err_block = torch.zeros_like(weight_block)
            losses_block = torch.zeros_like(weight_block)
            hessian_block = hessian_inv[i1:i2, i1:i2]

            for i in range(count):
                w = weight_block[:, i]
                d = hessian_block[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(weight[:, (i1 + i) : (i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize_tensor(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                quant_block[:, i] = q
                losses_block[:, i] = (w - q) ** 2 / d**2

                err = (w - q) / d
                weight_block[:, i:] -= err.unsqueeze(1).matmul(hessian_block[i, i:].unsqueeze(0))
                err_block[:, i] = err

            quantized_weight[:, i1:i2] = quant_block
            losses[:, i1:i2] = losses_block / 2
            weight[:, i2:] -= err_block.matmul(hessian_inv[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if actorder:
            quantized_weight = quantized_weight[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            quantized_weight = quantized_weight.t()
        self.layer.weight.data = quantized_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        quantized_params = int(self.layer.weight.numel())
        return {
            "time_s": round(time.time() - start, 4),
            "error": round(float(torch.sum(losses).item()), 6),
            "bits": int(round(math.log2(float(self.quantizer.maxq.item() + 1)))),
            "groupsize": int(groupsize),
            "quantized_params": quantized_params,
        }

    def free(self):
        self.hessian = None
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


def _module_selected(layer_idx: int, local_name: str, global_name: str, selected_targets: Set[str], gptq_config: Dict) -> bool:
    if global_name not in selected_targets:
        return False
    min_layer = gptq_config.get("min_layer", -1)
    max_layer = gptq_config.get("max_layer", 10**9)
    quant_only = gptq_config.get("quantize_only", "")
    invert = bool(gptq_config.get("invert", False))
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
        shift_mask = (
            labels_mask[:, 1:].contiguous().bool()
            if labels_mask is not None
            else torch.ones_like(shift_labels, dtype=torch.bool)
        )

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
def quantize_openpangu_sequential(model, input_ids, attention_mask, device, selected_targets: Set[str], gptq_config: Dict) -> Dict:
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
                gptq_config,
            )
        }
        execution_groups = build_execution_groups(list(all_linear.keys()), selected_local_names, bool(gptq_config.get("true_sequential", True)))

        for group in execution_groups:
            subset = {name: all_linear[name] for name in group}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    int(gptq_config["bits"]),
                    perchannel=True,
                    sym=bool(gptq_config.get("sym", False)),
                    mse=False,
                )

            def add_batch(name):
                def hook(_, inp, out):
                    gptq[name].add_batch(inp[0].detach(), out.detach())

                return hook

            handles = [module.register_forward_hook(add_batch(name)) for name, module in subset.items()]
            for sample_idx in range(hidden_states.shape[0]):
                next_hidden_states[sample_idx] = run_layer(layer, hidden_states[sample_idx], sample_kwargs[sample_idx], device)
            for handle in handles:
                handle.remove()

            for name in group:
                quant_metrics = gptq[name].fasterquant(
                    blocksize=int(gptq_config.get("block_size", 128)),
                    percdamp=float(gptq_config["damp_percent"]),
                    groupsize=int(gptq_config.get("group_size", -1)),
                    actorder=bool(gptq_config.get("desc_act", False)),
                    static_groups=bool(gptq_config.get("static_groups", False)),
                )
                weight = subset[name].weight.data
                total_params = int(weight.numel())
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
                gptq[name].free()

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
