# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Stochastic gradient descent constrained to the Stiefel manifold (orthonormal rows).

Each update projects the Euclidean gradient on the tangent space as a skew-symmetric matrix and
moves along it with a Cayley retraction, computed by a few fixed-point iterations. This is the
optimizer SpinQuant uses to learn rotations (Liu et al., "SpinQuant: LLM Quantization with
Learned Rotations").
# TODO(citation): add the final SpinQuant reference.
"""

import random

import torch
from torch import Tensor

EPS = 1e-8
FIXED_POINT_ITERATIONS = 5
REORTHONORMALIZE_ONE_IN = 101


def _orthonormalize_rows(rows: Tensor) -> Tensor:
    q, r = torch.linalg.qr(rows.t())
    q *= torch.diag(r, 0).sign().expand_as(q)
    return q.t()


def _tangent_skew(direction: Tensor, rows: Tensor) -> Tensor:
    """Skew-symmetric W = W_hat - W_hat^T with W_hat = D X - 1/2 X^T X D X (D: descent direction)."""
    d_x = torch.mm(direction, rows)
    x_d_x = torch.mm(rows, d_x)
    w_hat = d_x - 0.5 * torch.mm(rows.t(), x_d_x)
    return w_hat - w_hat.t()


def _cayley_retract(columns: Tensor, skew: Tensor, direction: Tensor, step) -> Tensor:
    """Fixed-point solution of Y = X + step * W (X + Y) / 2, started from X + step * D."""
    moved = columns + step * direction
    for _ in range(FIXED_POINT_ITERATIONS):
        moved = columns + step * torch.mm(skew, 0.5 * (columns + moved))
    return moved.t()


class StiefelSGD(torch.optim.Optimizer):
    """SGD for parameters with orthonormal rows (here: square rotation matrices).

    The step size is `lr`, capped at 1 / ||W||_1 so that the retraction stays accurate.
    """

    def __init__(self, params, lr: float):
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.copy_(self._moved(param, group["lr"]))
        return loss

    @staticmethod
    def _moved(param: Tensor, lr: float) -> Tensor:
        n_rows = param.size(0)
        rows = param.view(n_rows, -1)
        rows = rows / rows.norm(p=2, dim=1, keepdim=True).add(EPS)
        if rows.size(0) > rows.size(1):
            raise ValueError(f"StiefelSGD needs at most as many rows as columns, got {tuple(rows.shape)}.")
        if random.randint(1, REORTHONORMALIZE_ONE_IN) == 1:
            rows = _orthonormalize_rows(rows)

        gradient = param.grad.view(n_rows, -1)
        direction = torch.zeros(gradient.t().size(), device=gradient.device) - gradient.t()
        skew = _tangent_skew(direction, rows)
        step = min(1.0 / (skew.abs().sum(dim=0).max() + EPS), lr)
        return _cayley_retract(rows.t(), skew, direction, step).view(param.size())
