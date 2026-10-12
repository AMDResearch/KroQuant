# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""The low-rank stage (LoRaQ, applied after the Kronecker transforms).

Every selected layer gets a branch L R kept in the factor format, while the residual W - L R is
quantized like any weight. L and R start from the truncated SVD of the (transformed) weight and
are fitted in weight space (`LowRankFit`). The SVD at rank `svd_rank` is cached in `svd.pt`.
"""

from pathlib import Path

import torch
from loguru import logger
from torch import Tensor

from ..calibration.layerwise import LayerwiseStage
from ..hooks import HookSet
from ..models import Block, Layer, Pipeline, select_layers
from ..quant import Quantizer
from .branch import LowRankBranch
from .fit import LowRankFit
from .trainer import LoRaQLayerTrainer


def truncated_svd(weight: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """W ~ L R from the float64 SVD, singular values split evenly: L = U sqrt(S), R = sqrt(S) V^T."""
    u, s, vt = torch.linalg.svd(weight.to(torch.float64), full_matrices=False)
    root = torch.diag(torch.sqrt(s[:rank]))
    return u[:, :rank] @ root, root @ vt[:rank]


class LowRankStage(LayerwiseStage):
    results_name = "factors.pt"

    def __init__(self, name: str, layers: list[str], directory: Path, device: torch.device, *,
                 fmt: str, factor_format: str, rank: int, svd_rank: int, fit: dict):
        super().__init__(name, layers, directory, device)
        if svd_rank < rank:
            raise ValueError(f"The cached SVD rank ({svd_rank}) must be at least the branch rank ({rank}).")
        self.rank = rank
        self.svd_rank = svd_rank
        self.fit_settings = dict(fit)
        self.format = fmt
        self.weight_quantizer = Quantizer(fmt)
        self.factor_quantizer = Quantizer(factor_format)
        self.input_quantizer = Quantizer(fmt)
        self.svd: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}

    @property
    def svd_file(self) -> Path:
        return self.directory / "svd.pt"

    def before_layers(self, pipe: Pipeline) -> None:
        """Truncated SVD of every selected weight (as transformed by the earlier stages), cached."""
        self.svd = torch.load(self.svd_file, map_location="cpu", weights_only=False) if self.svd_file.exists() else {}
        for block in pipe.blocks:
            if block.name in self.svd:
                continue
            layers = select_layers(block, self.layers)
            if layers:
                self.svd[block.name] = {}
            for layer in layers:
                left, right = truncated_svd(layer.module.weight, self.svd_rank)
                self.svd[block.name][layer.name] = (left.detach().cpu(), right.detach().cpu())
            torch.save(self.svd, self.svd_file)
        logger.info(f"SVD of {sum(map(len, self.svd.values()))} layers in {self.svd_file}")

    def make_trainer(self, block: Block, layer: Layer) -> LoRaQLayerTrainer:
        left, right = self.svd[block.name][layer.name]
        fit = LowRankFit(residual_quantizer=Quantizer(self.format), factor_quantizer=Quantizer(self.format),
                         **self.fit_settings)
        return LoRaQLayerTrainer(
            fit=fit,
            layer=layer,
            l0=left[:, :self.rank],
            r0=right[:self.rank],
            weight_quantizer=self.weight_quantizer,
            factor_quantizer=self.factor_quantizer,
            input_quantizer=self.input_quantizer,
        )

    def attach(self, pipe: Pipeline) -> HookSet:
        """Quantize the stored factors again in the factor format on the layer's GPU, store the residual
        W - L R in the layer (in its dtype) and attach the branch."""
        hooks = HookSet()
        for block, layer, stored in self.stored_entries(pipe):
            left, right = stored["l"], stored["r"]
            if left.shape[-1] != self.rank or right.shape[0] != self.rank:
                raise ValueError(f"{layer.name}: stored factors of rank {left.shape[-1]}, expected {self.rank}.")
            weight = layer.module.weight.data
            with torch.no_grad():
                left, right = left.to(block.device), right.to(block.device)
                left = self.factor_quantizer(left)
                right = self.factor_quantizer(right.T).to(right).T
                branch = LowRankBranch(left, right, device=block.device, dtype=weight.dtype)
            weight.copy_(weight - branch.product())
            hooks.add(branch.attach(layer.module))
        return hooks
