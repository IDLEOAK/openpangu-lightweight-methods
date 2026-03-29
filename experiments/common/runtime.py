import os
from pathlib import Path
from typing import Optional, Tuple

import torch


def ensure_hf_home(hf_home: Optional[Path]) -> Path:
    if hf_home is None:
        hf_home = Path.cwd() / ".hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    return hf_home


def select_runtime() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def get_model_input_device(model, runtime_device: torch.device, device_map: str) -> torch.device:
    if device_map:
        return next(model.parameters()).device
    return runtime_device


def load_tokenizer_and_model(model_path: Path, device_map: str, hf_home: Path):
    ensure_hf_home(hf_home)
    runtime_device, model_dtype = select_runtime()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )

    model_load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
        "local_files_only": True,
    }
    if device_map:
        model_load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_load_kwargs)
    if not device_map:
        model.to(runtime_device)
    model.eval()

    return tokenizer, model, runtime_device, model_dtype
