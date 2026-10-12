# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""A light description of a diffusion transformer: its blocks and, inside each block, the linear
layers grouped by role. Layer names are full module paths of the transformer, e.g.
`transformer_blocks.3.attn1.to_q`, which the configs select with regexes."""

from dataclasses import dataclass, field
from typing import ClassVar, Optional

import torch
from torch import Tensor, nn


@dataclass(eq=False)
class Layer:
    name: str
    module: nn.Linear


@dataclass(eq=False)
class Attention:
    """Projections of one attention; `ctx_*` are the context-stream projections of joint attention."""

    q: Layer
    out: Layer
    k: Optional[Layer] = None
    v: Optional[Layer] = None
    ctx_q: Optional[Layer] = None
    ctx_k: Optional[Layer] = None
    ctx_v: Optional[Layer] = None
    ctx_out: Optional[Layer] = None

    def layers(self) -> list[Layer]:
        return [layer for layer in (self.q, self.k, self.v, self.out, self.ctx_q, self.ctx_k, self.ctx_v, self.ctx_out) if layer]


@dataclass(eq=False)
class MLP:
    up: Layer
    down: Layer


@dataclass(eq=False)
class Block:
    """One transformer block. Subclasses describe an architecture: which layers it has, how its
    call arguments split into per-sample inputs and constants (`split_inputs`), and how the outputs
    of the previous block become its per-sample inputs (`inputs_from`)."""

    name: str
    module: nn.Module
    attentions: list[Attention] = field(default_factory=list)
    mlps: list[MLP] = field(default_factory=list)
    norms: list[Layer] = field(default_factory=list)

    SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ("encoder_hidden_states", "hidden_states")

    def layers(self) -> list[Layer]:
        found = [layer for attention in self.attentions for layer in attention.layers()]
        found += [layer for mlp in self.mlps for layer in (mlp.up, mlp.down)]
        return found + list(self.norms)

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def find(self, path: str) -> Optional[Layer]:
        """The linear layer at `path` (relative to the block), or None if the block has none."""
        try:
            module = self.module.get_submodule(path)
        except AttributeError:
            return None
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{self.name}.{path} is a {type(module).__name__}, not a linear layer.")
        return Layer(f"{self.name}.{path}", module)

    def split_inputs(self, args: tuple, kwargs: dict) -> tuple[dict, dict]:
        """(per-sample inputs, constants) of a call: positional arguments are named after
        SAMPLE_INPUTS, the remaining SAMPLE_INPUTS are taken from the keywords, and every other
        keyword is a constant."""
        names = self.SAMPLE_INPUTS
        if len(args) > len(names):
            raise ValueError(f"{type(self).__name__} expects at most {len(names)} positional inputs, got {len(args)}.")
        per_sample = dict(zip(names, args))
        per_sample.update({name: kwargs[name] for name in names[len(args):] if name in kwargs})
        constants = {key: value for key, value in kwargs.items() if key not in per_sample}
        return per_sample, constants

    @classmethod
    def inputs_from(cls, outputs) -> dict:
        """Per-sample inputs from the outputs of the previous block: (context, hidden) states."""
        if not (isinstance(outputs, tuple) and len(outputs) == 2):
            raise ValueError(f"{cls.__name__} expects 2 outputs from the previous block, got {type(outputs).__name__}.")
        return dict(zip(cls.SAMPLE_INPUTS, outputs))


def single_tensor_inputs(block_class: type, outputs) -> dict:
    if not isinstance(outputs, Tensor):
        raise ValueError(f"{block_class.__name__} expects one tensor from the previous block, got {type(outputs).__name__}.")
    return {"hidden_states": outputs}
