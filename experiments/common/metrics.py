import math
import time
from typing import Dict, List

import torch

from experiments.common.runtime import get_model_input_device


def _render_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n{user_prompt}"


def _clean_response(output_sent: str) -> str:
    content = output_sent.strip()
    if "[unused17]" in content:
        content = content.split("[unused17]", maxsplit=1)[-1]
    if "[unused16]" in content:
        content = content.split("[unused16]")[-1]
    if "[unused10]" in content:
        content = content.split("[unused10]")[0]
    return content.strip()


def measure_generation(model, tokenizer, prompts: List[str], system_prompt: str, max_new_tokens: int, runtime_device, device_map: str) -> Dict:
    latencies = []
    token_counts = []
    samples = []
    input_device = get_model_input_device(model, runtime_device, device_map)

    with torch.inference_mode():
        for prompt in prompts:
            text = _render_chat(tokenizer, system_prompt, prompt)
            model_inputs = tokenizer([text], return_tensors="pt").to(input_device)
            start = time.perf_counter()
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id or 45892,
                return_dict_in_generate=True,
            )
            elapsed = time.perf_counter() - start
            input_length = model_inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            generated_count = int(generated_tokens.shape[1])
            output_sent = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)

            latencies.append(elapsed)
            token_counts.append(generated_count)
            samples.append(
                {
                    "prompt": prompt,
                    "generated_tokens": generated_count,
                    "latency_s": round(elapsed, 4),
                    "response": _clean_response(output_sent),
                }
            )

    total_tokens = sum(token_counts)
    total_latency = sum(latencies)
    return {
        "sample_count": len(prompts),
        "avg_latency_s": round(total_latency / max(len(latencies), 1), 4),
        "avg_generated_tokens": round(total_tokens / max(len(token_counts), 1), 2),
        "tokens_per_second": round(total_tokens / total_latency, 4) if total_latency > 0 else 0.0,
        "samples": samples,
    }


def measure_perplexity(model, tokenizer, texts: List[str], max_length: int, runtime_device, device_map: str) -> Dict:
    losses = []
    token_total = 0
    input_device = get_model_input_device(model, runtime_device, device_map)

    with torch.inference_mode():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_device)
            if input_ids.shape[1] < 2:
                continue
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = float(outputs.loss.detach().float().cpu())
            losses.append((loss, int(input_ids.shape[1] - 1)))
            token_total += int(input_ids.shape[1] - 1)

    if not losses or token_total == 0:
        return {"sample_count": 0, "avg_loss": None, "perplexity": None}

    weighted_loss = sum(loss * tokens for loss, tokens in losses) / token_total
    return {
        "sample_count": len(losses),
        "avg_loss": round(weighted_loss, 6),
        "perplexity": round(math.exp(weighted_loss), 6),
    }
