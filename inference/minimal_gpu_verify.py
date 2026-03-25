# coding=utf-8
"""
Minimal GPU verification script for openPangu-Embedded-7B.

What it verifies:
1) local tokenizer can be loaded
2) local model weights can be loaded
3) one short generation pass works on CUDA/CPU runtime
"""

import argparse
import os
import time
from pathlib import Path

import torch


def setup_hf_home(hf_home: str | None) -> str:
    if hf_home:
        target = Path(hf_home).resolve()
    else:
        target = Path(__file__).resolve().parents[1] / ".hf_cache"
    target.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(target)
    return str(target)


def pick_runtime() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def clean_response(output_sent: str) -> tuple[str, str]:
    thinking_content = ""
    content = output_sent.strip()
    if "[unused17]" in output_sent:
        thinking_content = output_sent.split("[unused17]")[0]
        content = output_sent.split("[unused17]", maxsplit=1)[-1]
    if "[unused16]" in thinking_content:
        thinking_content = thinking_content.split("[unused16]")[-1]
    if "[unused16]" in content:
        content = content.split("[unused16]")[-1]
    if "[unused10]" in content:
        content = content.split("[unused10]")[0]
    return thinking_content.strip(), content.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal runtime verification for openPangu-Embedded-7B")
    parser.add_argument("--model-path", type=str, default="", help="Local model directory. Default: repo root.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Short decode length for quick verification.")
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Transformers device_map. Use "auto" for safer low-VRAM verification, or empty string for single device.',
    )
    parser.add_argument("--hf-home", type=str, default="", help="HF cache path. Default: <repo>/.hf_cache")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Give me a one-sentence introduction to large language models.",
        help="User prompt for one short generation run.",
    )
    args = parser.parse_args()

    model_path = str(Path(args.model_path).resolve()) if args.model_path else str(Path(__file__).resolve().parents[1])
    hf_home = setup_hf_home(args.hf_home or None)
    runtime_device, model_dtype = pick_runtime()

    print(f"[INFO] model_path={model_path}")
    print(f"[INFO] hf_home={hf_home}")
    print(f"[INFO] runtime_device={runtime_device}, torch_dtype={model_dtype}")
    if torch.cuda.is_available():
        print(f"[INFO] cuda_device_count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[INFO] cuda:{i}={torch.cuda.get_device_name(i)}")
    print(f"[INFO] device_map={args.device_map!r}")

    # Import transformers after HF_HOME is set to avoid permission issues.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    t1 = time.time()

    model_load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=model_dtype,
        local_files_only=True,
    )
    device_map = args.device_map.strip()
    if device_map:
        model_load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)
    if not device_map:
        model.to(runtime_device)
    model.eval()
    t2 = time.time()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_device = next(model.parameters()).device if device_map else runtime_device
    model_inputs = tokenizer([text], return_tensors="pt").to(input_device)

    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id or 45892,
            return_dict_in_generate=True,
        )
    t3 = time.time()

    input_length = model_inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    output_sent = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
    thinking_content, content = clean_response(output_sent)

    print(f"[OK] tokenizer_load_s={t1 - t0:.2f}")
    print(f"[OK] model_load_s={t2 - t1:.2f}")
    print(f"[OK] generate_s={t3 - t2:.2f}")
    if thinking_content:
        print(f"[OK] thinking_content={thinking_content[:200]}")
    print(f"[OK] content={content[:300]}")
    print("[PASS] minimal verification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
