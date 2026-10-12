# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Per-layer trainer of the Kronecker stages: learns the transform of one linear layer from its
calibration inputs."""

import logging
import math
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from loguru import logger as log
from torch import Tensor
from tqdm import tqdm

from ..calibration.errors import evaluate_module_loss, evaluate_reference_loss
from ..models.tree import Layer
from ..quant import Quantizer
from ..quant.tensorcast import tcast
from .transform import TransformSpec

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


class STEFunction:
    """Quantize in the forward pass, identity gradient in the backward pass (straight-through)."""

    @staticmethod
    def apply(tensor: Tensor, quantizer: Quantizer) -> Tensor:
        dtype = quantizer.dtype

        quantized = tcast.cast(
            tensor,
            dtype=dtype,
        ).tensor
        return tensor + (quantized - tensor).detach()


class QuantizationLoss:
    """coef_o * MSE(output) + coef_w * MSE(weights) + coef_a * MSE(activations)."""

    def __init__(self, base_criterion=torch.nn.MSELoss(reduction='mean'), coef_o=1.0, coef_w=0.5, coef_a=0.5):
        assert coef_o >= 0 and coef_w >= 0 and coef_a >= 0, "Coefficients must be non-negative"
        self.base_criterion = base_criterion
        self.coef_o = coef_o
        self.coef_w = coef_w
        self.coef_a = coef_a

    def __call__(self, O_target, O_pred, A_target, A_pred, W_target, W_pred):
        loss_o = self.base_criterion(O_pred, O_target)
        loss_w = self.base_criterion(W_pred, W_target)
        loss_a = self.base_criterion(A_pred, A_target)

        total_loss = self.coef_o * loss_o + self.coef_w * loss_w + self.coef_a * loss_a
        return total_loss


class OutputOptimizer:
    """Learns the transform of one linear layer by minimizing the quantization error of its
    output, regularized by the weight and activation quantization errors (`QuantizationLoss`,
    with `coef_w`/`coef_a` set from the initial error ratios)."""

    def __init__(self,
        spec: TransformSpec,
        layer: Layer,
        input_quantizer: Quantizer,
        weight_quantizer: Quantizer,
        lr: float = 5e-3,
        epochs: int = 100,
        batch_size: int = 16,
    ):
        self._layer = layer
        weight: Tensor = layer.module.weight  # type: ignore
        self.device: torch.device = weight.device
        self.lr = lr
        self.criterion = QuantizationLoss()
        self.epochs = epochs
        self.input_quantizer = input_quantizer
        self.weight_quantizer = weight_quantizer
        self.batch_size = batch_size
        self.module_builder = spec

    @property
    def name(self) -> str:
        return self._layer.name

    @staticmethod
    def plot(result: Dict[str, Any], directory: str) -> None:
        """Loss curves of the training (output, weight and activation errors), saved in `directory`."""
        name = result.get('name', 'unknown')
        try:
            loss_history = result.get('loss_history', [])
            initial_output_loss = result.get('initial_output_loss', 0.0)
            reference_output_loss = result.get('reference_output_loss', 0.0)

            w_loss_history = result.get('w_loss_history', [])
            initial_weight_loss = result.get('initial_weight_loss', 0.0)
            reference_weight_loss = result.get('reference_weight_loss', 0.0)

            a_loss_history = result.get('a_loss_history', [])
            initial_activation_loss = result.get('initial_activation_loss', 0.0)
            reference_activation_loss = result.get('reference_activation_loss', 0.0)

            initial_fp_output_loss = result.get('initial_fp_output_loss', 0.0)
            initial_relative_output_loss = result.get('initial_relative_output_loss', 0.0)

            lr = result.get('lr', 0.0)

            os.makedirs(directory, exist_ok=True)

            safe_name = name.replace(os.sep, "_")
            fname = f"{safe_name}_lr{lr:.0e}_loss.png"
            out_path = os.path.join(directory, fname)

            fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

            axs[0].plot(range(1, len(loss_history) + 1), loss_history, color='black')
            axs[0].axhline(y=initial_output_loss, color='orange', linestyle='--', label=f'Initial Loss={initial_output_loss:.3e}')
            axs[0].axhline(y=reference_output_loss, color='red', linestyle='--', label=f'Reference Loss={reference_output_loss:.3e}')

            axs[0].plot([], [], ' ', label=f'Init FP Loss={initial_fp_output_loss:.3e}')
            axs[0].plot([], [], ' ', label=f'Init Rel Loss={initial_relative_output_loss:.3e}')

            axs[0].set_ylabel("Full loss")
            axs[0].set_title(f"Loss curve for {name} (lr={lr:.3e})")
            axs[0].set_yscale('log')
            axs[0].grid(True)
            if loss_history:
                axs[0].scatter(1, loss_history[0], color="orange", label=f"init={loss_history[0]:.3e}")
                best_idx = int(torch.argmin(torch.tensor(loss_history)).item())
                axs[0].scatter(best_idx + 1, loss_history[best_idx], color="green", label=f"best={loss_history[best_idx]:.3e}")
            axs[0].legend(loc="best", fontsize="small")

            axs[1].plot(range(1, len(w_loss_history) + 1), w_loss_history, color='purple')
            axs[1].axhline(y=initial_weight_loss, color='orange', linestyle='--', label=f'Initial W Loss={initial_weight_loss:.3e}')
            axs[1].axhline(y=reference_weight_loss, color='red', linestyle='--', label=f'Reference W Loss={reference_weight_loss:.3e}')
            axs[1].set_yscale('log')
            axs[1].set_ylabel("W Quant Error")
            axs[1].set_title("Weight Quantization Error W")
            axs[1].grid(True)
            if w_loss_history:
                axs[1].scatter(1, w_loss_history[0], color="orange", label=f"init={w_loss_history[0]:.3e}")
                best_w_idx = int(torch.argmin(torch.tensor(w_loss_history)).item())
                axs[1].scatter(best_w_idx + 1, w_loss_history[best_w_idx], color="green", label=f"min={w_loss_history[best_w_idx]:.3e}")
            axs[1].legend(loc="best", fontsize="small")

            axs[2].plot(range(1, len(a_loss_history) + 1), a_loss_history, color='brown')
            axs[2].axhline(y=initial_activation_loss, color='orange', linestyle='--', label=f'Initial A Loss={initial_activation_loss:.3e}')
            axs[2].axhline(y=reference_activation_loss, color='red', linestyle='--', label=f'Reference A Loss={reference_activation_loss:.3e}')

            axs[2].set_yscale('log')
            axs[2].set_xlabel("Epoch")
            axs[2].set_ylabel("A Quant Error")
            axs[2].set_title("Activation Quantization Error A")
            axs[2].grid(True)
            if a_loss_history:
                axs[2].scatter(1, a_loss_history[0], color="orange", label=f"init={a_loss_history[0]:.3e}")
                best_a_idx = int(torch.argmin(torch.tensor(a_loss_history)).item())
                axs[2].scatter(best_a_idx + 1, a_loss_history[best_a_idx], color="green", label=f"min={a_loss_history[best_a_idx]:.3e}")
            axs[2].legend(loc="best", fontsize="small")

            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            log.info(f"Saved loss plot to {out_path}")
        except Exception as e:
            log.error(f"Failed to save loss plot for {name}: {e}")

    def get_optimizer(self) -> tuple[torch.optim.Optimizer, ...]:
        return (torch.optim.Adam(self.kronecker.parameters(), lr=self.lr),)

    def _step(self, optimizers: tuple[torch.optim.Optimizer, ...], epoch: int):
        for opt in optimizers:
            opt.step()

    def train(self, inputs: Tensor) -> Dict[str, Any]:
        log.info(f"Optimizing Kronecker parameters for layer {self.name}.")
        input_shape = inputs.shape[-1]
        self.kronecker = self.module_builder.build(input_shape=input_shape, device=self.device)
        self.optimizers = self.get_optimizer()
        self.schedulers = tuple(
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
            for opt in self.optimizers
        )
        w = self._layer.module.weight.detach().clone().float()  # type: ignore
        a = inputs.detach().clone().float()

        n_samples = a.shape[0]

        best_state_dict = {k: v.detach().clone().cpu() for k, v in self.kronecker.state_dict().items()}

        best_loss = torch.inf
        initial_loss = -1.0

        epoch_losses: List[float] = []
        epoch_w_losses: List[float] = []
        epoch_a_losses: List[float] = []

        evals = {}

        with torch.no_grad():
            ref = a @ w.t()

        evals.update(
            evaluate_reference_loss(
                w,
                a,
                self.weight_quantizer,
                dtype=torch.float32,
                batch_size=self.batch_size
            )
        )

        evals.update(
            evaluate_module_loss(
                self.kronecker,
                w,
                a,
                self.weight_quantizer,
                dtype=torch.float32,
                batch_size=self.batch_size
            )
        )

        self.criterion.coef_w = 0.5 * 10 ** (math.ceil(math.log10(evals['initial_output_loss'] / evals['initial_weight_loss'])))
        self.criterion.coef_a = 0.5 * 10 ** (math.ceil(math.log10(evals['initial_output_loss'] / evals['initial_activation_loss'])))

        n_values_a = a.numel()
        n_values_ref = ref.numel()
        n_values_w = w.numel()

        log.info(f"coef_w: {self.criterion.coef_w}, coef_a: {self.criterion.coef_a}")

        with tqdm(range(self.epochs), desc=f"{self.name}", leave=False, position=1) as pbar:
            for epoch in pbar:
                perm = torch.randperm(n_samples, device=a.device)
                full_mse = 0.0
                full_a_loss = 0.0
                sum_loss = 0.0
                with torch.no_grad():
                    wt = self.kronecker(w, inverse=True)
                    error_w = (wt - STEFunction.apply(wt, self.weight_quantizer)).pow(2).sum().item() / n_values_w

                    for start in range(0, n_samples, self.batch_size):
                        idx = perm[start: start + self.batch_size]
                        a_batch = a[idx]
                        at_batch = self.kronecker(a_batch, inverse=False)
                        q_at_batch = STEFunction.apply(at_batch, self.input_quantizer)
                        error_a = (q_at_batch - at_batch).pow(2).sum().item()
                        full_a_loss += error_a
                    full_a_loss = full_a_loss / n_values_a

                for start in range(0, n_samples, self.batch_size):
                    idx = perm[start: start + self.batch_size]
                    at_batch = a[idx]
                    ref_batch = ref[idx]
                    wt = self.kronecker(w, inverse=True)

                    q_wt = STEFunction.apply(wt, self.weight_quantizer)
                    at_batch = self.kronecker(at_batch, inverse=False)
                    q_at_batch = STEFunction.apply(at_batch, self.input_quantizer)

                    pred = torch.matmul(q_at_batch, q_wt.T)

                    loss = self.criterion(
                        O_pred=pred, O_target=ref_batch,
                        W_pred=self.weight_quantizer(wt), W_target=wt,
                        A_pred=self.input_quantizer(at_batch), A_target=at_batch
                    )

                    with torch.no_grad():
                        full_mse += (pred - ref_batch).pow(2).sum().item()
                        sum_loss += loss.item()

                    for opt in self.optimizers:
                        opt.zero_grad()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.kronecker.parameters(), max_norm=1.0)
                    self._step(self.optimizers, epoch=epoch)

                for sched in self.schedulers:
                    sched.step()
                full_mse = full_mse / n_values_ref

                with torch.no_grad():
                    if initial_loss == -1.0:
                        initial_loss = full_mse

                    if sum_loss < best_loss:
                        best_loss = sum_loss
                        best_state_dict = {k: v.detach().clone().cpu() for k, v in self.kronecker.state_dict().items()}

                epoch_losses.append(full_mse)
                epoch_w_losses.append(error_w)
                epoch_a_losses.append(full_a_loss)
                pbar.set_postfix(
                    {
                        "init": f"{initial_loss:.3e}",
                        "current": f"{full_mse:.3e}",
                        "error_w": f"{error_w:.3e}",
                    }
                )

        result = {
            'config': self.module_builder.get_config(input_shape),
            'state_dict': best_state_dict,
            'loss_history': epoch_losses,
            'w_loss_history': epoch_w_losses,
            'a_loss_history': epoch_a_losses,
            'name': self.name,
            'lr': self.lr
        }
        result.update(evals)

        return result
