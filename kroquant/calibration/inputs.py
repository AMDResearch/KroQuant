# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Calibration inputs of the transformer blocks, and their propagation from block to block.

The full-precision pipeline denoises the calibration prompts; at every denoising step the call
arguments and outputs of the first block are recorded (the run stops after the last block of the
last step, without decoding images). A fixed random subset of the recorded samples is kept and
saved. Calibration then feeds them to one block at a time: the outputs of a block, with the
stages already calibrated installed, become the inputs of the next one.
"""

import math
import random
from pathlib import Path
from typing import Iterable, Iterator

import diffusers
import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from ..models import Block, Pipeline

Record = dict

SHARED_CONSTANTS = ("image_rotary_emb", "joint_attention_kwargs", "height", "width")


class _StopDenoising(Exception):
    """Raised by the last block once enough steps were recorded."""


def _tensors_to(record: Record, device, dtype: torch.dtype) -> Record:
    """Copy of `record` where every tensor (not those nested in tuples) is on `device` in `dtype`."""
    return {key: value.detach().to(device=device, dtype=dtype, copy=True) if isinstance(value, Tensor) else value
            for key, value in record.items()}


def _numbered(outputs) -> Record:
    """The outputs of a block as a record {"0": ..., "1": ...}."""
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    return {str(position): value.detach() for position, value in enumerate(outputs)}


def _chunks(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start: start + size]


def _first_dtype(records: list[Record]) -> torch.dtype:
    for record in records:
        for value in record.values():
            if isinstance(value, Tensor):
                return value.dtype
    raise ValueError("The records contain no tensor.")


def _concatenate(rows: list[Record], dtype: torch.dtype) -> Record:
    """Key-wise concatenation along the batch dimension, on the CPU, in `dtype` (None stays None)."""
    keys = list(rows[0])
    for row in rows[1:]:
        if set(row) != set(keys):
            raise KeyError(f"Records with different keys: {sorted(set(row) ^ set(keys))}.")
    merged = {}
    for key in keys:
        tensors = [row[key].to(device="cpu", dtype=dtype) for row in rows if row[key] is not None]
        merged[key] = torch.cat(tensors, dim=0) if tensors else None
    return merged


class BlockInputs:
    """The calibration samples, as the inputs of the block being calibrated.

    `inputs[i]`, `constants[i]` and `outputs[i]` describe batch `i`: per-sample inputs, per-step
    constants and outputs of the current block (the inputs of the next one); `shared` holds the
    constants common to all batches. Everything is kept on the CPU.
    """

    def __init__(self, path: Path, device: torch.device):
        self.path = Path(path)
        self.device = device
        self.inputs: list[Record] = []
        self.constants: list[Record] = []
        self.outputs: list[Record] = []
        self.shared: Record = {}

    def prepare(self, pipe: Pipeline, prompts: Iterable[dict], n_batches: int, num_samples: int, sample_seed: int,
                steps: int, guidance: float) -> None:
        """Load the saved samples, or record them with `pipe` (on the GPU) and save them."""
        if self.path.exists():
            saved = torch.load(self.path, map_location="cpu", weights_only=False)
            self.inputs, self.constants = saved["inputs"], saved["constants"]
            self.outputs, self.shared = saved["outputs"], saved["shared"]
            logger.info(f"Loaded {len(self.inputs)} batches of calibration inputs from {self.path}")
            return
        self._record(pipe, prompts, n_batches, steps, guidance)
        self._keep_subset(num_samples, sample_seed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"inputs": self.inputs, "constants": self.constants, "outputs": self.outputs, "shared": self.shared},
                   self.path)

    @torch.no_grad()
    def _record(self, pipe: Pipeline, prompts: Iterable[dict], n_batches: int, steps: int, guidance: float) -> None:
        first = pipe.blocks[0]
        last = pipe.blocks[-1] if steps > 1 else first
        steps = max(1, int(steps))

        def record(module, args, kwargs, outputs) -> None:
            dtype = next(module.parameters()).dtype
            per_sample, constants = first.split_inputs(args, kwargs)
            self.inputs.append(_tensors_to(per_sample, "cpu", dtype))
            self.constants.append(_tensors_to(constants, "cpu", dtype))
            self.outputs.append(_tensors_to(_numbered(outputs), "cpu", dtype))

        calls = 0

        def stop_after_last_step(*_) -> None:
            nonlocal calls
            calls += 1
            if calls >= steps:
                calls = 0
                raise _StopDenoising

        hooks = [first.module.register_forward_hook(record, with_kwargs=True),
                 last.module.register_forward_hook(stop_after_last_step, prepend=False)]
        try:
            for batch in tqdm(prompts, total=n_batches, desc="Recording calibration inputs", leave=False):
                torch.manual_seed(0)
                diffusers.training_utils.set_seed(0)
                generators = [torch.Generator(self.device).manual_seed(int(seed)) for seed in batch["seed"]]
                try:
                    with torch.no_grad():
                        pipe(prompt=batch["prompt"], num_images_per_prompt=1, num_inference_steps=steps,
                             guidance_scale=guidance, generator=generators, output_type="latent")
                except _StopDenoising:
                    pass
                finally:
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()
        finally:
            for hook in hooks:
                hook.remove()

    def _keep_subset(self, num_samples: int, seed: int) -> None:
        """Keep `num_samples` recorded samples, chosen at random, re-batched by at most 64.

        Sample `i` of the shuffled order is row `i % batch size` of batch `i % number of batches`.
        """
        total = sum(next(iter(record.values())).shape[0] for record in self.inputs)
        order = list(range(total))
        random.Random(seed).shuffle(order)
        order = order[:num_samples]
        batch_size = min(math.ceil(num_samples), 64)

        first_constants = self.constants[0]
        for key in SHARED_CONSTANTS:
            value = first_constants.get(key, None)
            if value is None:
                continue
            if key == "joint_attention_kwargs" and not (isinstance(value, dict) and not value):
                raise ValueError("Non-empty joint_attention_kwargs cannot be shared by all samples.")
            self.shared[key] = value
            for record in self.constants:
                record.pop(key, None)

        def resample(records: list[Record]) -> list[Record]:
            dtype = _first_dtype(records)
            batches = []
            for chunk in _chunks(order, batch_size):
                rows = []
                for index in chunk:
                    source = records[index % len(records)]
                    rows.append({key: value[index % value.shape[0]: index % value.shape[0] + 1] if value is not None else value
                                 for key, value in source.items()})
                batches.append(_concatenate(rows, dtype))
            return batches

        self.inputs = resample(self.inputs)
        self.constants = resample(self.constants)
        self.outputs = resample(self.outputs)

    def enter(self, block: Block) -> None:
        """Make the stored outputs (of the previous block) the inputs of `block`."""
        logger.info(f"Calibration inputs now feed {block.name}")
        converted = []
        for outputs in self.outputs:
            values = tuple(outputs.values())
            per_sample = block.inputs_from(values[0] if len(outputs) == 1 else values)
            converted.append({key: value.detach().to("cpu") if isinstance(value, Tensor) else value
                              for key, value in per_sample.items()})
        self.inputs = converted

    def run(self, block: Block, keep_outputs: bool) -> None:
        """Call `block` on every batch (its hooks see the inputs); with `keep_outputs`, its outputs
        replace the stored ones."""
        dtype = next(block.module.parameters()).dtype
        for index in tqdm(range(len(self.inputs)), desc=f"Running {block.name}", leave=False):
            arguments = _tensors_to(self.inputs[index], self.device, dtype)
            arguments.update(_tensors_to(self.constants[index], self.device, dtype))
            arguments.update(_tensors_to(self.shared, self.device, dtype))
            try:
                with torch.no_grad():
                    outputs = block.module(**arguments)
            except Exception as error:
                raise RuntimeError(f"{block.name} failed on calibration batch {index}") from error
            if keep_outputs:
                self.outputs[index] = _tensors_to(_numbered(outputs), "cpu", dtype)
