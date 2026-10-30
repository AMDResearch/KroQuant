# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Modules swapped into the transformers so that every quantized projection is an `nn.Linear`.

Both build their linear layers with the regular initialization before copying the weights in.
The models are wrapped on the CPU, so this consumes the global CPU generator, in block order:
the random draws of the later stages depend on it.
"""

import torch
import torch.nn as nn


class SplitLinear(nn.Module):
    """A linear layer over concatenated inputs, as one linear per input chunk:

        y = sum_i x_i W_i^T + b,   x = [x_1, ..., x_k] split along the features.

    FLUX single-stream blocks project the concatenation of the attention and MLP outputs with
    one layer (`proj_out`); split, each part reads (and quantizes) its own input. The bias is
    carried by the last chunk.
    """

    def __init__(self, chunk_sizes: list[int], out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        if len(chunk_sizes) < 2:
            raise ValueError("SplitLinear needs at least 2 input chunks.")
        self.chunk_sizes = list(chunk_sizes)
        self.in_features = sum(chunk_sizes)
        self.out_features = out_features
        last = len(chunk_sizes) - 1
        self.linears = nn.ModuleList(
            nn.Linear(size, out_features, bias and index == last, device, dtype) for index, size in enumerate(chunk_sizes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.split(self.chunk_sizes, dim=-1)
        return sum(linear(chunk.contiguous()) for linear, chunk in zip(self.linears, chunks, strict=True))

    @classmethod
    def split(cls, linear: nn.Linear, leading_sizes: list[int]) -> "SplitLinear":
        """`linear` cut along its input features: chunks of `leading_sizes`, then the rest."""
        sizes = [size for size in [*leading_sizes, linear.in_features - sum(leading_sizes)] if size > 0]
        parts = cls(sizes, linear.out_features, bias=linear.bias is not None,
                    device=linear.weight.device, dtype=linear.weight.dtype)
        offset = 0
        for part in parts.linears:
            part.weight.data.copy_(linear.weight[:, offset: offset + part.in_features])
            offset += part.in_features
        if linear.bias is not None:
            parts.linears[-1].bias.data.copy_(linear.bias)
        return parts


class PointwiseConv(nn.Module):
    """A 1x1 convolution computed by a linear layer over the channels (SANA's MLP)."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (1, 1):
            raise ValueError("PointwiseConv replaces 1x1 nn.Conv2d layers only.")
        dtype, device = conv.weight.dtype, conv.weight.device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.linear = nn.Linear(self.in_channels, self.out_channels, bias=conv.bias is not None, dtype=dtype, device=device)
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.squeeze(-1).squeeze(-1))
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).view(batch, height * width, channels)
        out = self.linear(tokens)
        return out.view(batch, height, width, self.out_channels).permute(0, 3, 1, 2)
