# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""The Kronecker-structured block transform of KroQuant, and how it is attached to a linear layer."""

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..hooks import HookSet

EPSILON = 1e-12


class InvertibleTransform(nn.Module):
    """An invertible linear transform M of the input features of a linear layer.

    Attached to a layer y = x W^T, it computes (x M)(W M^-T)^T = x W^T, so the layer is
    unchanged in full precision while its inputs and weights are quantized in the
    transformed basis.
    """

    def __init__(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self._fallback_device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fallback_dtype = dtype if dtype else torch.float32

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            try:
                return next(self.buffers()).dtype
            except StopIteration:
                return self._fallback_dtype

    @dtype.setter
    def dtype(self, new_dtype: torch.dtype):
        self._fallback_dtype = new_dtype
        self.to(dtype=new_dtype)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return self._fallback_device

    @device.setter
    def device(self, new_device: torch.device):
        self._fallback_device = new_device
        self.to(device=new_device)

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        raise NotImplementedError

    def _transform_inputs(self, module: nn.Module, inputs: tuple):
        if not inputs:
            return None
        return (self.forward(inputs[0], inverse=False), *inputs[1:])

    def attach(self, layer: nn.Module) -> HookSet:
        """Apply M^-T to the weight of `layer` in place (computed in float32), then move the transform
        to the layer dtype and apply M to its inputs (a pre-hook after those already registered)."""
        weight = layer.weight
        if not isinstance(weight, Tensor):
            raise ValueError("The layer must have a 'weight' tensor.")
        target_dtype = weight.dtype
        self.device = weight.device
        with torch.no_grad():
            transformed = self.forward(weight.data.to(self.device, dtype=self.dtype), inverse=True)
            weight.data.copy_(transformed.to(target_dtype))
        self.dtype = target_dtype
        return HookSet(layer.register_forward_pre_hook(self._transform_inputs, prepend=False, with_kwargs=False))


class KroneckerTransform(InvertibleTransform):
    """Block-diagonal invertible transform with Kronecker-structured blocks.

    The input features are split in blocks of size 2**N. Block i is
    M_i = S_i (G_i1 (x) G_i2 (x) ... (x) G_iN), with S_i a random +-1 diagonal (if `random_init`)
    and 2x2 factors G = L U of determinant -1:

        L = [[1, 0], [c, 1]],   U = [[a, b], [0, -1/a]],   learnable (a, b, c) per factor.

    At initialization (a, b, c) = (cos, sin, tan)(pi/4), so G = H_2 / sqrt(2) and each block is
    the randomized Hadamard matrix S_i H_{2**N} / sqrt(2**N): with `epochs: 0` the transform is a
    block Hadamard rotation.

    `forward(x)` applies M to the activations and `forward(W, inverse=True)` applies M^-T to the
    weights, so that the layer output is unchanged in full precision.
    """

    def __init__(self, input_shape, N: int = 5, random_init=False, device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        self.n_parameters = N
        self.size = 2 ** N
        assert input_shape % self.size == 0, f"Shape {input_shape} is not divisible by Kronecker size {self.size}."
        self.shape = input_shape // self.size

        self.register_buffer('base_l', torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]], device=self.device, dtype=self.dtype))
        self.register_buffer('base_u_a', torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], device=self.device, dtype=self.dtype))
        self.register_buffer('base_u_b', torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]], device=self.device, dtype=self.dtype))
        self.register_buffer('base_u_d', torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]], device=self.device, dtype=self.dtype))

        self.random_init = random_init
        self._init_params()

    def _init_params(self):
        if self.random_init:
            signs = torch.randint(0, 2, (self.shape, self.size), device=self.device, dtype=self.dtype)
            self.register_buffer('random_diag', torch.diag_embed(2 * signs - 1))
        k = torch.randint(0, 1, (self.n_parameters, self.shape, 1, 1), device=self.device).to(dtype=self.dtype)
        theta = (math.pi / 4) + k * (math.pi / 2)

        params = torch.zeros(self.n_parameters, self.shape, 1, 1, 3, device=self.device, dtype=self.dtype)
        params[..., 0] = torch.cos(theta)
        params[..., 1] = torch.sin(theta)
        params[..., 2] = torch.tan(theta)
        self._params = torch.nn.Parameter(params)

    @property
    def a(self):
        return self._params[..., 0]

    @property
    def b(self):
        return self._params[..., 1]

    @property
    def c(self):
        return self._params[..., 2]

    def _factors(self):
        """The 2x2 factors G = L U, shape (N, blocks, 2, 2)."""
        a_safe = self.a + EPSILON * torch.sign(self.a)
        d_safe = -1.0 / a_safe

        L = torch.eye(2, device=self.device, dtype=self.dtype) + self.c * self.base_l  # type: ignore
        U = a_safe * self.base_u_a + self.b * self.base_u_b + d_safe * self.base_u_d  # type: ignore
        return L @ U

    def _inverse_factors(self):
        """The inverses G^-1 = U^-1 L^-1, in closed form."""
        a_safe = 1.0 / (self.a + EPSILON * torch.sign(self.a))
        d_safe = -1.0 / a_safe
        b_safe = -self.b * a_safe * d_safe

        L = torch.eye(2, device=self.device, dtype=self.dtype) - self.c * self.base_l  # type: ignore
        U = a_safe * self.base_u_a + b_safe * self.base_u_b + d_safe * self.base_u_d  # type: ignore
        return U @ L

    def _batch_kronecker(self, x, y):
        *_, x_a, x_b = x.shape
        *_, y_a, y_b = y.shape
        siz1 = torch.Size([x_a * y_a, x_b * y_b])
        res = x.unsqueeze(-1).unsqueeze(-3) * y.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)

    def _kronecker_product(self, inverse=False):
        factors = self._inverse_factors() if inverse else self._factors()
        krons = factors[0]
        for i in range(1, self.n_parameters):
            krons = self._batch_kronecker(krons, factors[i])
        return krons

    @property
    def matrix(self) -> Tensor:
        """The diagonal blocks M_i, shape (blocks, 2**N, 2**N)."""
        kron = self._kronecker_product()
        if self.random_init:
            return self.random_diag @ kron
        return kron

    @property
    def inverse_matrix(self) -> Tensor:
        """The inverses M_i^-1."""
        kron_inverse = self._kronecker_product(inverse=True)
        if self.random_init:
            return kron_inverse @ self.random_diag
        return kron_inverse

    def _transform(self, x: Tensor) -> Tensor:
        *other_dims, n = x.shape
        X = x.view(*other_dims, -1, self.size)
        if len(other_dims) == 1:
            return torch.bmm(X.permute(1,0,2), self.matrix).permute(1,0,2).reshape(*other_dims, n)
        elif len(other_dims) == 2:
            return torch.einsum('nmik, mkj->nmij', X.permute(0,2,1,3), self.matrix).permute(0,2,1,3).reshape(*other_dims, n)
        raise NotImplementedError("Only supports up to 2 batch dimensions.")

    def _inverse_transform(self, x: Tensor) -> Tensor:
        *other_dims, n = x.shape
        X = x.view(*other_dims, -1, self.size)
        if len(other_dims) == 1:
            return torch.bmm(X.permute(1,0,2), self.inverse_matrix.transpose(-2, -1)).permute(1,0,2).reshape(*other_dims, n)
        elif len(other_dims) == 2:
            return torch.einsum('nmik, mjk->nmij', X.permute(0,2,1,3), self.inverse_matrix).permute(0,2,1,3).reshape(*other_dims, n)
        raise NotImplementedError("Only supports up to 2 batch dimensions.")

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        if inverse:
            return self._inverse_transform(x)
        return self._transform(x)


@dataclass(frozen=True)
class TransformSpec:
    """How the Kronecker transform of a layer is built: `N` 2x2 factors per block (blocks of
    2**N features), with or without the random sign diagonal."""

    N: int = 5
    random_init: bool = True

    def build(self, input_shape: int, device: torch.device = torch.device("cuda:0")) -> KroneckerTransform:
        return KroneckerTransform(input_shape=input_shape, device=device, N=self.N, random_init=self.random_init)

    def get_config(self, input_shape: int) -> dict[str, Any]:
        """Stored with the results, to rebuild the transform."""
        return {"class_name": KroneckerTransform.__name__, "input_shape": input_shape, "N": self.N, "random_init": self.random_init}
