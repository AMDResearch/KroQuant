# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Loading a diffusers pipeline and describing its transformer for calibration."""

from typing import Iterator

import diffusers
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .blocks import FluxJointBlock, FluxSingleBlock, PixArtBlock, SanaBlock
from .tree import Block, Layer

ARCHITECTURES: dict[str, list[tuple[str, type[Block]]]] = {
    "PixArtTransformer2DModel": [("transformer_blocks", PixArtBlock)],
    "SanaTransformer2DModel": [("transformer_blocks", SanaBlock)],
    "FluxTransformer2DModel": [("transformer_blocks", FluxJointBlock), ("single_transformer_blocks", FluxSingleBlock)],
}


def describe_blocks(transformer: nn.Module) -> list[Block]:
    """The blocks of a supported transformer, in execution order (module surgery included)."""
    kind = type(transformer).__name__
    if kind not in ARCHITECTURES:
        raise ValueError(f"Unsupported transformer {kind}; supported: {', '.join(ARCHITECTURES)}.")
    blocks: list[Block] = []
    for attribute, block_class in ARCHITECTURES[kind]:
        for index, module in getattr(transformer, attribute).named_children():
            blocks.append(block_class(f"{attribute}.{index}", module))
    return blocks


def load_diffusers(model: DictConfig):
    """The diffusers pipeline of the model config, on the CPU."""
    pipeline_class = getattr(diffusers, model.pipeline)
    extra = OmegaConf.to_container(model.load_kwargs, resolve=True) if model.get("load_kwargs") else {}
    return pipeline_class.from_pretrained(model.repo, use_safetensors=True, torch_dtype=getattr(torch, model.torch_dtype), **extra)


class Pipeline:
    """A diffusers pipeline together with the description of its transformer blocks."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.blocks: list[Block] = describe_blocks(pipe.transformer)

    @classmethod
    def load(cls, model: DictConfig) -> "Pipeline":
        return cls(load_diffusers(model))

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)

    def to(self, device) -> "Pipeline":
        self.pipe.to(device)
        return self

    def layers(self) -> Iterator[Layer]:
        for block in self.blocks:
            yield from block.layers()

    def silence_progress_bars(self) -> None:
        self.pipe.set_progress_bar_config(disable=True)
