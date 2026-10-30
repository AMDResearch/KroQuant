# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Fake quantization (quantize, then dequantize back to the input dtype) with TensorCast."""

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .formats import number_format
from .tensorcast import tcast


class Quantizer:
    """Rounds a tensor to the number format `fmt`, block by block along its last dimension.

    TensorCast stores per-call state on the `DataType` object, so two threads must not share a
    quantizer: give each worker its own `copy()`.
    """

    def __init__(self, fmt: str) -> None:
        self.format = fmt
        self.dtype = number_format(fmt)

    def __call__(self, x: Tensor) -> Tensor:
        return tcast.cast(x, self.dtype).tensor

    def copy(self) -> "Quantizer":
        return Quantizer(self.format)

    def __repr__(self) -> str:
        return f"Quantizer({self.format!r})"


def quantize_inputs(module: nn.Module, quantizer: Quantizer) -> RemovableHandle:
    """Quantize the tensor arguments of every call to `module` (a forward pre-hook, run after the
    pre-hooks already registered on it)."""

    def pre_hook(_: nn.Module, args: tuple) -> tuple:
        return tuple(quantizer(arg) if isinstance(arg, Tensor) else arg for arg in args)

    return module.register_forward_pre_hook(pre_hook)


@torch.no_grad()
def quantize_weight_(module: nn.Module, quantizer: Quantizer) -> None:
    """Replace the weight of `module` by its quantized value, in place, keeping its dtype."""
    weight = module.weight.data
    weight.copy_(quantizer(weight).to(weight.dtype))
