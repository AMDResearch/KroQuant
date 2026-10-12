# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Weight-space fit of a low-rank pair (L, R) for a weight W, in two phases.

1. Adam on (L, R) so that the residual W - L R quantizes well: loss ||E - Q(E)||^2, E = W - L R.
2. For each matrix of the batch, an orthogonal r (initialized at a randomized Hadamard matrix and
   updated by `StiefelSGD`) so that L r and r^T R quantize well; L R is unchanged by r.
"""

from dataclasses import dataclass

import torch
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm

from ..quant import Quantizer
from .cayley import StiefelSGD
from .hadamard import random_orthogonal_hadamard


def _squared_error() -> nn.Module:
    return nn.MSELoss(reduction="sum")


def _initial_rotation(left: Tensor) -> nn.Parameter:
    size = left.shape[1]
    try:
        start = random_orthogonal_hadamard(size, left.device).to(left.dtype)
    except ValueError:
        logger.warning(f"No Hadamard matrix of order {size}: the rotation starts from the identity.")
        start = torch.eye(size, device=left.device, dtype=left.dtype)
    return nn.Parameter(start, requires_grad=True)


@dataclass
class LowRankFit:
    """Hyperparameters of the fit; calling it fits a batch of (W, L0, R0) and returns (L, R).

    `residual_quantizer` quantizes W - L R in phase 1, `factor_quantizer` quantizes L and R^T in
    phase 2.
    """

    residual_quantizer: Quantizer
    factor_quantizer: Quantizer
    lr: float = 1e-3
    epochs: int = 500
    rotation_lr: float = 0.5
    rotation_epochs: int = 500

    def __call__(self, weight: Tensor, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        """`weight`: (batch, out, in); `left`: (batch, out, rank); `right`: (batch, rank, in)."""
        target = weight.to(torch.float32)
        left = nn.Parameter(left.to(torch.float32))
        right = nn.Parameter(right.to(torch.float32))
        self._fit_residual(target, left, right)
        pairs = [self._fit_rotation(left[i], right[i]) for i in tqdm(range(left.shape[0]), leave=False, desc="rotations")]
        return torch.stack([pair[0] for pair in pairs], dim=0), torch.stack([pair[1] for pair in pairs], dim=0)

    def _fit_residual(self, target: Tensor, left: nn.Parameter, right: nn.Parameter) -> None:
        adam = torch.optim.Adam([left, right], lr=self.lr)
        error = _squared_error()
        progress = tqdm(range(self.epochs), leave=False, desc="residual")
        for _ in progress:
            adam.zero_grad()
            residual = target - left @ right
            loss = error(residual, self.residual_quantizer(residual).to(residual))
            loss.backward()
            adam.step()
            progress.set_postfix(loss=f"{loss.item():.4e}")

    def _fit_rotation(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        rotation = _initial_rotation(left)
        left, right = left.detach(), right.detach()
        sgd = StiefelSGD([rotation], lr=self.rotation_lr)
        error = _squared_error()
        quantize = self.factor_quantizer
        progress = tqdm(range(self.rotation_epochs), leave=False, desc="rotation")
        for _ in progress:
            sgd.zero_grad()
            rotated_left, rotated_right = torch.mm(left, rotation), torch.mm(rotation.T, right)
            quantized_left = quantize(rotated_left).to(rotated_left)
            quantized_right = quantize(rotated_right.T).T.to(rotated_right)
            loss = error(quantized_left, rotated_left) + error(quantized_right, rotated_right)
            loss.backward()
            sgd.step()
            progress.set_postfix(loss=f"{loss.item():.4e}")
        with torch.no_grad():
            return torch.mm(left, rotation), torch.mm(rotation.T, right)
