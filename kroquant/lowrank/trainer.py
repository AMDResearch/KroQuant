# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Per-layer trainer of the low-rank stage (LoRaQ), with input-aware diagnostics."""

import os
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from loguru import logger as log
from matplotlib.lines import Line2D
from torch import Tensor

from ..calibration.errors import evaluate_reference_loss
from ..models.tree import Layer
from ..quant import Quantizer
from .fit import LowRankFit


def _reversed_dims(tensor: Tensor) -> Tensor:
    """All dimensions in reverse order (`tensor.T`, also for more than two dimensions)."""
    return tensor.permute(*reversed(range(tensor.dim())))


class LoRaQLayerTrainer:
    """Fits the low-rank branch (L, R) of one layer, starting from its truncated SVD, with the
    weight-space fit `fit` (`LowRankFit`). The calibration inputs of the layer are only used to
    report the output error before and after (the diagnostics stored with the factors)."""

    def __init__(
        self,
        fit: LowRankFit,
        layer: Layer,
        l0: Tensor,
        r0: Tensor,
        weight_quantizer: Quantizer,
        factor_quantizer: Quantizer,
        input_quantizer: Quantizer,
        batch_size: int = 32,
    ):
        self.fit = fit
        self.layer = layer
        self.w_quantizer = weight_quantizer
        self.lr_quantizer = factor_quantizer
        self.i_quantizer = input_quantizer
        self.batch_size = batch_size

        self.l = l0.float().clone().to(layer.module.weight.device)
        self.r = r0.float().clone().to(layer.module.weight.device)
        log.debug(f"{layer.name}: L {tuple(self.l.shape)}, R {tuple(self.r.shape)}")

        self.device = layer.module.weight.device

    @property
    def name(self) -> str:
        return self.layer.name

    def _left_cast(self, mat: Tensor) -> Tensor:
        return self.lr_quantizer(mat)

    def _right_cast(self, mat: Tensor) -> Tensor:
        return _reversed_dims(self.lr_quantizer(_reversed_dims(mat)).to(mat))

    def train(self, inputs: Tensor) -> Dict[str, Any]:
        w = self.layer.module.weight.detach().clone().float()
        inputs = inputs.to(self.device).float()
        n_samples = inputs.shape[0]

        evals = evaluate_reference_loss(
            w,
            inputs,
            self.w_quantizer,
            dtype=torch.float32,
        )

        with torch.no_grad():
            ref_out = inputs @ w.t()
            n_values_ref = ref_out.numel()

        l, r = self.fit(w.unsqueeze(0), self.l.detach().clone().unsqueeze(0), self.r.detach().clone().unsqueeze(0))

        l, r = self._left_cast(l).squeeze(0), self._right_cast(r).squeeze(0)

        residual = self.w_quantizer(w - (l @ r)).to(w)

        final_output_loss = 0.0
        for start in range(0, n_samples, self.batch_size):
            idx = slice(start, min(start + self.batch_size, n_samples))
            x_batch = inputs[idx]
            y_ref_batch = ref_out[idx]
            with torch.no_grad():
                qx_batch = self.i_quantizer(x_batch).to(x_batch)
                y_pred = (x_batch @ r.T) @ l.T + (qx_batch @ residual.T)
                final_output_loss += (y_pred - y_ref_batch).pow(2).sum().item()
        final_output_loss /= n_values_ref

        result = {
            "l": l.detach().cpu(),
            "r": r.detach().cpu(),
            "final_output_loss": final_output_loss,
            "name": self.name,
            "lr": self.fit.lr,
        }
        result.update(evals)
        return result

    @staticmethod
    def plot(result: Dict[str, Any], directory: str) -> None:
        """Bar plot of the output error before and after, saved in `directory`."""
        name = result.get("name", "unknown")
        try:
            initial_loss = result["reference_output_loss"]
            initial_activation_loss = result["reference_activation_loss"]
            initial_weight_loss = result["reference_weight_loss"]

            final_loss = result["final_output_loss"]
            lr = result.get("lr", 0.0)

            os.makedirs(directory, exist_ok=True)

            safe_name = name.replace(os.sep, "_")
            fname = f"{safe_name}_lr{lr:.0e}_loss.png"
            out_path = os.path.join(directory, fname)

            plt.figure()
            bars = plt.bar(["Initial Output Loss", "Final Output Loss"], [initial_loss, final_loss], color=["blue", "green"], width=0.4)

            handles = [
                bars[0],
                bars[1],
                Line2D([0], [0], color="none", label="Init Act"),
                Line2D([0], [0], color="none", label="Init Weight"),
            ]
            labels = [
                f"Initial: {initial_loss:.2e}",
                f"Final: {final_loss:.2e}",
                f"Init Act: {initial_activation_loss:.2e}",
                f"Init W: {initial_weight_loss:.2e}",
            ]

            plt.legend(handles, labels)
            plt.ylabel("MSE Loss")
            plt.yscale("log")
            plt.title(f"Output Loss Before and After Optimization for {name}")
            plt.savefig(out_path)
            plt.close()
            log.info(f"Saved loss plot for {name}")
        except Exception as e:
            log.error(f"Failed to save loss plot for {name}: {e}")
