import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from safetensors.torch import load_file, save_file

from experiments.common.reporting import summarize_directory


def pack_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    flat = flat.view(-1, 8)
    shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    return torch.sum(flat * shifts.unsqueeze(0), dim=1).to(torch.uint8)


def unpack_bool_mask(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed = packed.to(torch.uint8).reshape(-1)
    bits = ((packed.unsqueeze(1) >> torch.arange(8, dtype=torch.uint8)) & 1).reshape(-1)
    return bits[:numel].to(torch.bool)


def pack_nbit_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return codes.reshape(-1).to(torch.uint8)
    if 8 % bits != 0:
        raise ValueError(f"Unsupported bits={bits}")
    values_per_byte = 8 // bits
    flat = codes.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % values_per_byte
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    flat = flat.view(-1, values_per_byte)
    packed = torch.zeros(flat.shape[0], dtype=torch.uint8)
    mask = (1 << bits) - 1
    for idx in range(values_per_byte):
        packed |= (flat[:, idx] & mask) << (idx * bits)
    return packed


def unpack_nbit_codes(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    packed = packed.reshape(-1).to(torch.uint8)
    if bits == 8:
        return packed[:count].to(torch.uint8)
    if 8 % bits != 0:
        raise ValueError(f"Unsupported bits={bits}")
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    chunks = []
    for idx in range(values_per_byte):
        chunks.append(((packed >> (idx * bits)) & mask).to(torch.uint8))
    return torch.stack(chunks, dim=1).reshape(-1)[:count]


def quant_group_count(columns: int, group_size: int) -> int:
    effective_group_size = columns if group_size <= 0 else group_size
    return int(math.ceil(columns / effective_group_size))


def export_sparse_artifact(
    model,
    target_module_names: Iterable[str],
    output_dir: Path,
    method: str,
    base_model_path_hint: str,
    source_model_path: str,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    named_modules = dict(model.named_modules())
    tensors = {}
    manifest_modules = []
    total_nonzero = 0
    total_params = 0

    for module_name in sorted(set(target_module_names)):
        module = named_modules[module_name]
        weight = module.weight.data.detach().cpu().contiguous()
        flat = weight.reshape(-1)
        mask = flat != 0
        packed_mask = pack_bool_mask(mask)
        values = flat[mask]
        total_nonzero += int(values.numel())
        total_params += int(flat.numel())

        mask_key = f"{module_name}.mask"
        values_key = f"{module_name}.values"
        tensors[mask_key] = packed_mask
        tensors[values_key] = values
        manifest_modules.append(
            {
                "name": module_name,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype).replace("torch.", ""),
                "numel": int(flat.numel()),
                "nonzero_count": int(values.numel()),
                "mask_key": mask_key,
                "values_key": values_key,
            }
        )

    tensor_path = output_dir / "artifact.safetensors"
    save_file(tensors, tensor_path)
    manifest = {
        "format_version": 1,
        "artifact_type": "sparse_overlay",
        "method": method,
        "storage_format": "mask_plus_values",
        "base_model_required": True,
        "base_model_path_hint": base_model_path_hint,
        "source_model_path": source_model_path,
        "target_module_count": len(manifest_modules),
        "total_target_params": total_params,
        "total_nonzero_params": total_nonzero,
        "modules": manifest_modules,
        "tensor_file": tensor_path.name,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = summarize_directory(output_dir)
    info["manifest"] = manifest
    return info


def export_quant_artifact(
    artifact_payloads: Dict[str, Dict],
    output_dir: Path,
    method: str,
    base_model_path_hint: str,
    source_model_path: str,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    manifest_modules = []
    total_params = 0

    for module_name in sorted(artifact_payloads):
        payload = artifact_payloads[module_name]
        packed_key = f"{module_name}.packed"
        scales_key = f"{module_name}.scales"
        zeros_key = f"{module_name}.zeros"
        tensors[packed_key] = payload["packed_codes"].detach().cpu()
        tensors[scales_key] = payload["scales"].detach().cpu()
        tensors[zeros_key] = payload["zeros"].detach().cpu()
        pre_scale_key = None
        if payload.get("pre_scale") is not None:
            pre_scale_key = f"{module_name}.pre_scale"
            tensors[pre_scale_key] = payload["pre_scale"].detach().cpu()

        rows, cols = payload["shape"]
        total_params += int(rows * cols)
        manifest_modules.append(
            {
                "name": module_name,
                "shape": list(payload["shape"]),
                "dtype": payload["dtype"],
                "bits": int(payload["bits"]),
                "group_size": int(payload["group_size"]),
                "sym": bool(payload.get("sym", False)),
                "packed_codes_key": packed_key,
                "scales_key": scales_key,
                "zeros_key": zeros_key,
                "pre_scale_key": pre_scale_key,
            }
        )

    tensor_path = output_dir / "artifact.safetensors"
    save_file(tensors, tensor_path)
    manifest = {
        "format_version": 1,
        "artifact_type": "packed_quant_overlay",
        "method": method,
        "storage_format": "packed_nbit_plus_scales",
        "base_model_required": True,
        "base_model_path_hint": base_model_path_hint,
        "source_model_path": source_model_path,
        "target_module_count": len(manifest_modules),
        "total_target_params": total_params,
        "modules": manifest_modules,
        "tensor_file": tensor_path.name,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = summarize_directory(output_dir)
    info["manifest"] = manifest
    return info


def _assign_weight(module, weight: torch.Tensor, dtype_name: str) -> None:
    target_dtype = getattr(torch, dtype_name)
    module.weight.data = weight.reshape(module.weight.shape).to(device=module.weight.device, dtype=target_dtype)


def load_sparse_artifact(model, artifact_dir: Path) -> Dict:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    tensors = load_file(str(artifact_dir / manifest["tensor_file"]))
    named_modules = dict(model.named_modules())

    for module_info in manifest["modules"]:
        module = named_modules[module_info["name"]]
        numel = int(module_info["numel"])
        shape = tuple(module_info["shape"])
        mask = unpack_bool_mask(tensors[module_info["mask_key"]], numel)
        values = tensors[module_info["values_key"]]
        dense = torch.zeros(numel, dtype=values.dtype)
        dense[mask] = values
        _assign_weight(module, dense.reshape(shape), module_info["dtype"])
    return manifest


def load_quant_artifact(model, artifact_dir: Path) -> Dict:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    tensors = load_file(str(artifact_dir / manifest["tensor_file"]))
    named_modules = dict(model.named_modules())

    for module_info in manifest["modules"]:
        module = named_modules[module_info["name"]]
        rows, cols = module_info["shape"]
        bits = int(module_info["bits"])
        group_size = int(module_info["group_size"])
        sym = bool(module_info.get("sym", False))
        maxq = 2**bits - 1

        packed = tensors[module_info["packed_codes_key"]]
        codes = unpack_nbit_codes(packed, bits, rows * cols).reshape(rows, cols).to(torch.float32)
        scales = tensors[module_info["scales_key"]].to(torch.float32)
        zeros = tensors[module_info["zeros_key"]].to(torch.float32)
        pre_scale = tensors[module_info["pre_scale_key"]].to(torch.float32) if module_info.get("pre_scale_key") else None

        effective_group_size = cols if group_size <= 0 else group_size
        restored = torch.zeros((rows, cols), dtype=torch.float32)
        group_count = quant_group_count(cols, group_size)
        for group_idx in range(group_count):
            start = group_idx * effective_group_size
            end = min(start + effective_group_size, cols)
            scale = scales[:, group_idx].unsqueeze(1)
            zero = zeros[:, group_idx].unsqueeze(1)
            block = scale * (codes[:, start:end] - zero)
            if pre_scale is not None:
                block = block / pre_scale[start:end].unsqueeze(0)
            restored[:, start:end] = block
        _assign_weight(module, restored, module_info["dtype"])
    return manifest
