"""A scoped Ultralytics YAML parser for the DSG-YOLO26 layer set.

Only the operations used by ``configs/dsg-yolo26.yaml`` are accepted. Keeping this
parser explicit avoids changing ``ultralytics.nn.tasks`` or its global module
registry. The channel/repeat/scaling rules mirror Ultralytics 8.4's YOLO26 parser.
"""

from __future__ import annotations

import ast
from typing import Any

import torch
from torch import nn
from ultralytics.nn.modules import C2PSA, C3k2, Concat, Conv, Detect, SPPF
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import make_divisible

from .modules import DPSE, SGCF, DetailPreservingShallowEnhancement, ScaleGatedCrossLevelFusion

SUPPORTED_SCALES = ("n", "s", "m", "l", "x")

_MODULE_REGISTRY = {
    "C2PSA": C2PSA,
    "C3k2": C3k2,
    "Concat": Concat,
    "Conv": Conv,
    "DPSE": DPSE,
    "Detect": Detect,
    "DetailPreservingShallowEnhancement": DetailPreservingShallowEnhancement,
    "SGCF": SGCF,
    "SPPF": SPPF,
    "ScaleGatedCrossLevelFusion": ScaleGatedCrossLevelFusion,
}

_CHANNEL_MODULES = frozenset({Conv, C3k2, C2PSA, SPPF, DetailPreservingShallowEnhancement})
_REPEAT_MODULES = frozenset({C3k2, C2PSA, DetailPreservingShallowEnhancement})


def _resolve_module(name: str) -> type[nn.Module]:
    """Resolve a supported built-in, custom, or torch layer by name."""
    if name.startswith("nn."):
        module = getattr(nn, name[3:], None)
    else:
        module = _MODULE_REGISTRY.get(name)
    if not isinstance(module, type) or not issubclass(module, nn.Module):
        supported = ", ".join(sorted(_MODULE_REGISTRY))
        raise ValueError(f"Unsupported DSG-YOLO26 layer '{name}'. Supported custom/built-in names: {supported}")
    return module


def _evaluate_args(args: list[Any], symbols: dict[str, Any]) -> list[Any]:
    """Evaluate YAML strings using the same safe, local-symbol rules as Ultralytics."""
    evaluated = []
    for arg in args:
        if isinstance(arg, str):
            if arg in symbols:
                arg = symbols[arg]
            else:
                try:
                    arg = ast.literal_eval(arg)
                except (ValueError, SyntaxError):
                    pass
        evaluated.append(arg)
    return evaluated


def parse_dsg_model(d: dict[str, Any], ch: int, verbose: bool = True) -> tuple[nn.Sequential, list[int]]:
    """Parse a DSG-YOLO26 configuration into a sequential PyTorch network.

    Args:
        d (dict): Model dictionary loaded from YAML.
        ch (int): Input image channels.
        verbose (bool): Print one row per network layer.

    Returns:
        tuple[nn.Sequential, list[int]): Network and indices whose outputs must be saved.
    """
    nc = d.get("nc", 80)
    end2end = d.get("end2end", False)
    reg_max = d.get("reg_max", 16)
    scales = d.get("scales")
    scale = d.get("scale", "")
    depth = d.get("depth_multiple", 1.0)
    width = d.get("width_multiple", 1.0)
    max_channels = float("inf")

    if d.get("activation"):
        raise ValueError("The scoped DSG parser does not support global activation overrides")
    if scales:
        if not scale:
            scale = next(iter(scales))
            LOGGER.warning(f"No model scale passed. Assuming scale='{scale}'.")
        if scale not in scales:
            raise ValueError(f"Unknown model scale '{scale}'; choose one of {tuple(scales)}")
        depth, width, max_channels = scales[scale]
    if scale and scale not in SUPPORTED_SCALES:
        raise ValueError(f"Unknown YOLO scale '{scale}'; expected one of {SUPPORTED_SCALES}")

    def scale_channels(value: int) -> int:
        return make_divisible(min(value, max_channels) * width, 8)

    symbols = {"nc": nc, "scale": scale, "depth": depth, "width": width}
    channels = [ch]
    layers: list[nn.Module] = []
    save: list[int] = []
    legacy = True
    c2 = channels[-1]

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    for index, (source, repeats, module_name, raw_args) in enumerate(d["backbone"] + d["head"]):
        module = _resolve_module(module_name)
        args = _evaluate_args(list(raw_args), symbols)
        repeats = repeats_for_log = max(round(repeats * depth), 1) if repeats > 1 else repeats

        if module in _CHANNEL_MODULES:
            if not isinstance(source, int):
                raise TypeError(f"Layer {index} ({module_name}) expects one input, got {source}")
            c1, c2 = channels[source], args[0]
            c2 = scale_channels(c2)
            args = [c1, c2, *args[1:]]
            if module in _REPEAT_MODULES:
                args.insert(2, repeats)
                repeats = 1
            if module is C3k2:
                legacy = False
                if scale in {"m", "l", "x"}:
                    args[3:4] = [True]
        elif module is ScaleGatedCrossLevelFusion:
            if not isinstance(source, list):
                raise TypeError(f"Layer {index} ({module_name}) expects a list of inputs, got {source}")
            c2 = scale_channels(args[0])
            args = [[channels[x] for x in source], c2, *args[1:]]
        elif module is Concat:
            if not isinstance(source, list):
                raise TypeError(f"Layer {index} (Concat) expects a list of inputs, got {source}")
            c2 = sum(channels[x] for x in source)
        elif module is Detect:
            if not isinstance(source, list):
                raise TypeError(f"Layer {index} (Detect) expects a list of inputs, got {source}")
            args = [*args, reg_max, end2end, [channels[x] for x in source]]
            Detect.legacy = legacy
            c2 = channels[source[0]]
        elif module in {nn.Upsample, nn.Identity}:
            c2 = channels[source]
        else:  # Defensive: all registry entries should be covered above.
            raise ValueError(f"No channel rule implemented for layer '{module_name}' at index {index}")

        instance = nn.Sequential(*(module(*args) for _ in range(repeats))) if repeats > 1 else module(*args)
        instance.np = sum(parameter.numel() for parameter in instance.parameters())
        instance.i, instance.f, instance.type = index, source, f"{module.__module__}.{module.__name__}"
        if verbose:
            LOGGER.info(
                f"{index:>3}{source!s:>20}{repeats_for_log:>3}{instance.np:10.0f}  "
                f"{instance.type:<45.45}{args!s:<30.30}"
            )
        save.extend(x % index for x in ([source] if isinstance(source, int) else source) if x != -1)
        layers.append(instance)
        if index == 0:
            channels = []
        channels.append(c2)

    return nn.Sequential(*layers), sorted(save)
