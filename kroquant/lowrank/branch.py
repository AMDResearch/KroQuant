# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""The low-rank branch added to a linear layer whose weight holds the residual W - L R."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..hooks import HookSet


class LowRankBranch(nn.Module):
    """x -> (x R^T) L^T, added to the output of the layer it is attached to.

    `left` is L (out_features x rank) and `right` is R (rank x in_features); both are stored in
    the layer's device and dtype.
    """

    def __init__(self, left: Tensor, right: Tensor, device: torch.device, dtype: torch.dtype):
        super().__init__()
        if left.shape[1] != right.shape[0]:
            raise ValueError(f"Rank mismatch between L {tuple(left.shape)} and R {tuple(right.shape)}.")
        self.register_buffer("left", left.to(device=device, dtype=dtype))
        self.register_buffer("right", right.to(device=device, dtype=dtype))
        self._branch_output: Tensor | None = None

    def product(self) -> Tensor:
        """L R, computed in the layer dtype."""
        return self.left @ self.right

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.right), self.left)

    def attach(self, layer: nn.Module) -> HookSet:
        """The branch reads the layer input as seen after the pre-hooks already on the layer (the
        transforms of the earlier stages); its output is added before any other forward hook."""
        return HookSet(
            layer.register_forward_pre_hook(self._read_input, prepend=False),
            layer.register_forward_hook(self._add_output, prepend=True),
        )

    def _read_input(self, _: nn.Module, args: tuple) -> None:
        if self._branch_output is not None:
            raise RuntimeError("The low-rank branch received a second input before its output was used.")
        self._branch_output = self(args[0])

    def _add_output(self, _: nn.Module, args: tuple, output):
        if self._branch_output is None:
            raise RuntimeError("The low-rank branch output is missing.")
        if isinstance(output, tuple):
            output = output[0]
        total = output + self._branch_output
        self._branch_output = None
        return total
