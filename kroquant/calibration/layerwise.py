# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Layer-by-layer calibration, shared by the Kronecker stages and the low-rank stage.

The calibration inputs go through the blocks one at a time, with the earlier stages installed.
While a block runs, pre-hooks record the inputs of its selected layers (as transformed by the
earlier stages); each layer is then calibrated on its own inputs. Results are saved after every
block, and a rerun skips the layers already calibrated.
"""

from pathlib import Path
from typing import Any, Iterable

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from ..hooks import HookSet
from ..models import Block, Layer, Pipeline, select_layers
from .inputs import BlockInputs


class _InputRecorder:
    """Keeps the first argument of every call to a layer (pre-hook after the existing ones)."""

    def __init__(self, layer: Layer):
        self.layer = layer
        self.calls: list[Tensor] = []
        self._handle = layer.module.register_forward_pre_hook(self._keep, prepend=False)

    def _keep(self, _, args: tuple) -> None:
        self.calls.append(args[0].detach())

    def stop(self) -> None:
        self._handle.remove()

    def inputs(self) -> Tensor:
        return torch.cat(self.calls, dim=0)


class LayerwiseStage:
    """A calibration stage whose result is computed independently for every selected layer.

    Subclasses provide `make_trainer` (an object with `train(inputs) -> dict` and a static
    `plot(result, directory)`), `results_file` and `attach`.
    """

    results_name = "results.pt"

    def __init__(self, name: str, layers: list[str], directory: Path, device: torch.device):
        self.name = name
        self.layers = list(layers)
        self.directory = Path(directory)
        self.device = device
        self.results: dict[str, dict[str, Any]] = {}

    @property
    def results_file(self) -> Path:
        return self.directory / self.results_name

    def make_trainer(self, block: Block, layer: Layer):
        raise NotImplementedError

    def before_layers(self, pipe: Pipeline) -> None:
        """Called with the whole pipeline on the GPU, before the calibration inputs are prepared."""

    def load_results(self) -> dict[str, dict[str, Any]]:
        if not self.results_file.exists():
            return {}
        return torch.load(self.results_file, map_location="cpu", weights_only=False)

    def save_results(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.results, self.results_file)

    def calibrate(self, pipe: Pipeline, inputs: BlockInputs, prompts: Iterable[dict], n_batches: int,
                  num_samples: int, sample_seed: int, steps: int, guidance: float) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.results = self.load_results()
        if self.results:
            logger.info(f"Resuming {self.name}: {sum(map(len, self.results.values()))} layers already calibrated")
        self.before_layers(pipe)
        inputs.prepare(pipe, prompts, n_batches, num_samples, sample_seed, steps, guidance)

        pipe.to("cpu")
        torch.cuda.empty_cache()
        for index, block in enumerate(tqdm(pipe.blocks, desc=f"{self.name}: blocks")):
            block.module.to(self.device)
            if index > 0:
                inputs.enter(block)
            done = self.results.get(block.name, {})
            recorders = [_InputRecorder(layer) for layer in select_layers(block, self.layers) if layer.name not in done]
            try:
                inputs.run(block, keep_outputs=index > 0)
            finally:
                for recorder in recorders:
                    recorder.stop()

            trained = []
            for recorder in recorders:
                logger.info(f"{self.name}: calibrating {recorder.layer.name}")
                trainer = self.make_trainer(block, recorder.layer)
                trained.append((recorder.layer, trainer, trainer.train(recorder.inputs())))
                recorder.calls.clear()
            for layer, trainer, result in trained:
                self.results.setdefault(block.name, {})[layer.name] = result
                trainer.plot(result, str(self.directory / "plots"))
            if trained:
                self.save_results()
            torch.cuda.empty_cache()
        self.save_results()
        logger.info(f"{self.name}: calibrated {sum(map(len, self.results.values()))} layers")

    def attach(self, pipe: Pipeline) -> HookSet:
        raise NotImplementedError

    def stored_entries(self, pipe: Pipeline):
        """(block, layer, stored result) for every selected layer that has a result."""
        results = self.load_results()
        if not results:
            raise FileNotFoundError(f"No results in {self.results_file}: run step={self.name} first.")
        for block in pipe.blocks:
            if block.name not in results:
                logger.warning(f"{self.name}: no results for {block.name}, skipped")
                continue
            for layer in select_layers(block, self.layers):
                if layer.name not in results[block.name]:
                    logger.warning(f"{self.name}: no results for {layer.name}, skipped")
                    continue
                yield block, layer, results[block.name][layer.name]
