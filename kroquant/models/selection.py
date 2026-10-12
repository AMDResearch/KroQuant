# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Which layers a stage works on: those whose name fully matches one of its regexes."""

import re
from typing import Iterable, Sequence

from .tree import Block, Layer


def matches(name: str, patterns: Iterable[str]) -> bool:
    return any(re.fullmatch(pattern, name) for pattern in patterns)


def calibration_order(block: Block) -> list[Layer]:
    """Every layer of `block`, in the order in which the stages process them: for each attention,
    query/key/value, then the context query/key/value, the output and the context output
    projections; then each MLP (up, down); then the modulation projections."""
    order: list[Layer | None] = []
    for attention in block.attentions:
        order += [attention.q, attention.k, attention.v, attention.ctx_q, attention.ctx_k, attention.ctx_v,
                  attention.out, attention.ctx_out]
    for mlp in block.mlps:
        order += [mlp.up, mlp.down]
    order += block.norms
    return [layer for layer in order if layer is not None]


def select_layers(block: Block, patterns: Sequence[str]) -> list[Layer]:
    return [layer for layer in calibration_order(block) if matches(layer.name, patterns)]
