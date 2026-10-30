# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""The transformer blocks of the supported architectures (diffusers 0.34)."""

from typing import ClassVar

import torch
from loguru import logger
from torch import Tensor, nn

from .surgery import PointwiseConv, SplitLinear
from .tree import MLP, Attention, Block, single_tensor_inputs


def _attention(block: Block, prefix: str, out: str | None = None, joint: bool = False) -> Attention:
    q, o = block.find(f"{prefix}.to_q"), block.find(out or f"{prefix}.to_out.0")
    if q is None or o is None:
        raise ValueError(f"{block.name}: attention {prefix!r} without query or output projection.")
    attention = Attention(q=q, out=o, k=block.find(f"{prefix}.to_k"), v=block.find(f"{prefix}.to_v"))
    if joint:
        attention.ctx_q = block.find(f"{prefix}.add_q_proj")
        attention.ctx_k = block.find(f"{prefix}.add_k_proj")
        attention.ctx_v = block.find(f"{prefix}.add_v_proj")
        attention.ctx_out = block.find(f"{prefix}.to_add_out")
    return attention


def _mlp(block: Block, up: str, down: str) -> MLP:
    up_layer, down_layer = block.find(up), block.find(down)
    if up_layer is None or down_layer is None:
        raise ValueError(f"{block.name}: MLP without {up!r} or {down!r}.")
    return MLP(up=up_layer, down=down_layer)


class PixArtBlock(Block):
    """PixArt-Sigma: self-attention, cross-attention, MLP. The hidden states are the only
    positional argument; every keyword argument is a constant of the step."""

    SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ("hidden_states",)

    def __init__(self, name: str, module: nn.Module):
        super().__init__(name, module)
        self.attentions = [_attention(self, "attn1"), _attention(self, "attn2")]
        self.mlps = [_mlp(self, "ff.net.0.proj", "ff.net.2")]

    def split_inputs(self, args: tuple, kwargs: dict) -> tuple[dict, dict]:
        if len(args) != 1:
            raise ValueError(f"PixArt blocks take the hidden states as their only positional argument, got {len(args)}.")
        return {"hidden_states": args[0]}, dict(kwargs)

    @classmethod
    def inputs_from(cls, outputs) -> dict:
        return single_tensor_inputs(cls, outputs)


class SanaBlock(Block):
    """SANA: self-attention, cross-attention and a convolutional MLP (GLUMBConv), whose 1x1
    convolutions are replaced by linear layers. All arguments are positional."""

    SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ("hidden_states",)
    CONSTANTS: ClassVar[tuple[str, ...]] = (
        "attention_mask", "encoder_hidden_states", "encoder_attention_mask", "timestep", "height", "width",
    )

    def __init__(self, name: str, module: nn.Module):
        module.ff.conv_inverted = PointwiseConv(module.ff.conv_inverted)
        module.ff.conv_point = PointwiseConv(module.ff.conv_point)
        super().__init__(name, module)
        self.attentions = [_attention(self, "attn1"), _attention(self, "attn2")]
        self.mlps = [_mlp(self, "ff.conv_inverted.linear", "ff.conv_point.linear")]

    def split_inputs(self, args: tuple, kwargs: dict) -> tuple[dict, dict]:
        if len(args) != 1 + len(self.CONSTANTS):
            raise ValueError(f"SANA blocks take {1 + len(self.CONSTANTS)} positional arguments, got {len(args)}.")
        hidden_states, *constants = args
        return {"hidden_states": hidden_states}, dict(zip(self.CONSTANTS, constants))

    @classmethod
    def inputs_from(cls, outputs) -> dict:
        return single_tensor_inputs(cls, outputs)


class FluxJointBlock(Block):
    """FLUX double-stream block: joint attention over the context and image streams, one MLP per
    stream, and the modulation projections of both streams."""

    def __init__(self, name: str, module: nn.Module):
        super().__init__(name, module)
        self.attentions = [_attention(self, "attn", joint=True)]
        self.mlps = [_mlp(self, "ff.net.0.proj", "ff.net.2"), _mlp(self, "ff_context.net.0.proj", "ff_context.net.2")]
        self.norms = [layer for layer in (self.find("norm1.linear"), self.find("norm1_context.linear")) if layer]


class FluxSingleBlock(Block):
    """FLUX single-stream block: attention and MLP run in parallel on the concatenated streams,
    and their outputs are projected together by `proj_out`, split here in two linear layers
    (attention part, MLP part)."""

    SAMPLE_INPUTS: ClassVar[tuple[str, ...]] = ("hidden_states",)

    def __init__(self, name: str, module: nn.Module):
        logger.info(f"Splitting proj_out of {name} into attention and MLP parts.")
        module.proj_out = SplitLinear.split(module.proj_out, [module.proj_out.out_features])
        super().__init__(name, module)
        self.attentions = [_attention(self, "attn", out="proj_out.linears.0")]
        self.mlps = [_mlp(self, "proj_mlp", "proj_out.linears.1")]
        self.norms = [layer for layer in (self.find("norm.linear"),) if layer]

    @classmethod
    def inputs_from(cls, outputs) -> dict:
        """After the last double-stream block, the two streams are concatenated (context first)."""
        if isinstance(outputs, Tensor):
            return {"hidden_states": outputs}
        if not isinstance(outputs, tuple) or len(outputs) not in (1, 2):
            raise ValueError(f"FluxSingleBlock expects 1 or 2 outputs from the previous block, got {type(outputs).__name__}.")
        if len(outputs) == 1:
            return {"hidden_states": outputs[0]}
        context, hidden = outputs
        return {"hidden_states": torch.cat([context, hidden], dim=1)}
