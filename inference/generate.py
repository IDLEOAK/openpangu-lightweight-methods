# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid permission issues from default user-level cache locations.
if "HF_HOME" not in os.environ:
    default_hf_home = REPO_ROOT / ".hf_cache"
    default_hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(default_hf_home)

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_LOCAL_PATH = os.getenv("MODEL_PATH", str(REPO_ROOT))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
MODEL_DEVICE_MAP = os.getenv("MODEL_DEVICE_MAP", "").strip()


def select_runtime() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


runtime_device, model_dtype = select_runtime()
print(f"[INFO] model_path={MODEL_LOCAL_PATH}")
print(f"[INFO] hf_home={os.environ['HF_HOME']}")
print(f"[INFO] runtime_device={runtime_device}, torch_dtype={model_dtype}")
if MODEL_DEVICE_MAP:
    print(f"[INFO] model_device_map={MODEL_DEVICE_MAP}")

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_LOCAL_PATH,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True,
)

model_load_kwargs = dict(
    trust_remote_code=True,
    torch_dtype=model_dtype,
    local_files_only=True,
)
if MODEL_DEVICE_MAP:
    model_load_kwargs["device_map"] = MODEL_DEVICE_MAP

model = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL_PATH, **model_load_kwargs)
if not MODEL_DEVICE_MAP:
    model.to(runtime_device)
model.eval()

# prepare the model input
sys_prompt = "You are a helpful assistant."
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_device = next(model.parameters()).device if MODEL_DEVICE_MAP else runtime_device
model_inputs = tokenizer([text], return_tensors="pt").to(input_device)

# conduct text completion
with torch.inference_mode():
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id or 45892,
        return_dict_in_generate=True,
    )

input_length = model_inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
output_sent = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)

# parsing thinking content
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

print("\nthinking content:", thinking_content.strip())
print("\ncontent:", content.strip())
