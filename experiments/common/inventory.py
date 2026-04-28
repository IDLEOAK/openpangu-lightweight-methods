from collections import Counter
from typing import Dict, List

import torch.nn as nn


def _classify_module(name: str) -> str:
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def collect_linear_inventory(model) -> Dict:
    modules = []
    counts = Counter()
    total_params = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        param_count = module.weight.numel()
        if module.bias is not None:
            param_count += module.bias.numel()
        group = _classify_module(name)
        counts[group] += 1
        total_params += param_count
        modules.append(
            {
                "name": name,
                "group": group,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "has_bias": module.bias is not None,
                "parameter_count": int(param_count),
            }
        )

    return {
        "total_linear_modules": len(modules),
        "total_linear_parameters": int(total_params),
        "group_counts": dict(counts),
        "modules": modules,
    }


def select_target_modules(inventory: Dict, include_groups: List[str], exclude_patterns: List[str]) -> List[Dict]:
    selected = []
    for module in inventory["modules"]:
        if include_groups and module["group"] not in include_groups:
            continue
        if any(pattern in module["name"] for pattern in exclude_patterns):
            continue
        selected.append(module)
    return selected


def validate_target_modules(
    inventory: Dict,
    selected: List[Dict],
    include_groups: List[str],
    exclude_patterns: List[str],
) -> None:
    available_groups = sorted({module["group"] for module in inventory["modules"]})
    unknown_groups = sorted(set(include_groups) - set(available_groups))
    if unknown_groups:
        raise ValueError(
            f"Unknown include_groups={unknown_groups}. Available groups={available_groups}"
        )

    if not selected:
        raise ValueError(
            "Target module selection is empty. "
            f"include_groups={include_groups}, exclude_patterns={exclude_patterns}, "
            f"available_groups={available_groups}"
        )
