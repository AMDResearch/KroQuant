# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Hadamard matrices, the starting point of the learned rotation of the low-rank factors."""

import torch

_H2 = ((1.0, 1.0), (1.0, -1.0))


def hadamard(order: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The Sylvester-Hadamard matrix of a power-of-two `order`: entries +-1, H_2k = H_2 (x) H_k."""
    if order < 1 or order & (order - 1):
        raise ValueError(f"Hadamard matrices are built for power-of-two orders only, got {order}.")
    matrix = torch.ones((1, 1), dtype=dtype)
    step = torch.tensor(_H2, dtype=dtype)
    while matrix.shape[0] < order:
        matrix = torch.kron(step, matrix)
    return matrix


def random_orthogonal_hadamard(order: int, device=None) -> torch.Tensor:
    """diag(s) H / sqrt(order) with independent random signs s, in float64.

    The signs come from the global CPU generator and are drawn before the size is validated
    (a non-power-of-two `order` still consumes them, then raises `ValueError`).
    """
    signs = torch.randint(low=0, high=2, size=(order,)).to(torch.float64) * 2 - 1
    norm = torch.tensor(order).sqrt()
    return (signs.unsqueeze(1) * hadamard(order)).div(norm).to(device)
