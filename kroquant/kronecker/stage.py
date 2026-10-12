# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""The Kronecker stages: one learned `KroneckerTransform` per selected layer.

`kronecker_init` uses 0 epochs: the transforms keep their randomized Hadamard initialization.
"""

from pathlib import Path

import torch

from ..calibration.layerwise import LayerwiseStage
from ..hooks import HookSet
from ..models import Block, Layer, Pipeline
from ..quant import Quantizer
from .trainer import OutputOptimizer
from .transform import TransformSpec


class KroneckerStage(LayerwiseStage):
    results_name = "transforms.pt"

    def __init__(self, name: str, layers: list[str], directory: Path, device: torch.device, *,
                 fmt: str, spec: TransformSpec, lr: float, epochs: int, batch_size: int):
        super().__init__(name, layers, directory, device)
        self.format = fmt
        self.spec = spec
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def make_trainer(self, block: Block, layer: Layer) -> OutputOptimizer:
        return OutputOptimizer(
            spec=self.spec,
            layer=layer,
            input_quantizer=Quantizer(self.format),
            weight_quantizer=Quantizer(self.format),
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

    def attach(self, pipe: Pipeline) -> HookSet:
        """Rebuild every stored transform (drawing its random initialization again, on the layer's
        GPU, before loading the stored state) and attach it to its layer."""
        hooks = HookSet()
        for block, layer, stored in self.stored_entries(pipe):
            transform = self.spec.build(input_shape=stored["config"]["input_shape"], device=block.device)
            transform.load_state_dict(stored["state_dict"])
            hooks.add(transform.attach(layer.module))
        return hooks
