# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Quantization errors of a linear layer on its calibration inputs (mean squared errors of the
weights, the activations and the outputs), with or without a transform."""

import torch


@torch.no_grad()
def evaluate_reference_loss(w, a, quantizer, dtype=torch.float32, batch_size=64):
    """Errors of the plain quantization (no transform): weights, activations and outputs."""
    assert w.dtype == dtype, f"Weights must be in {dtype} for evaluation."
    assert a.dtype == dtype, f"Activations must be in {dtype} for evaluation."

    n_samples = a.shape[0]

    n_values_a = a.numel()
    n_values_w = w.numel()

    o = a @ w.t()
    n_values_o = o.numel()

    q_w = quantizer(w)

    reference_weight_loss = (w - q_w).pow(2).sum().item()
    reference_activation_loss = 0.0
    reference_output_loss = 0.0

    for i in range(0, n_samples, batch_size):
        idx = slice(i, min(i + batch_size, n_samples))
        a_batch = a[idx]
        o_batch = o[idx]

        q_a_batch = quantizer(a_batch)
        reference_activation_loss += (a_batch - q_a_batch).pow(2).sum().item()
        reference_output_loss += (o_batch - (q_a_batch @ q_w.t())).pow(2).sum().item()

    reference_output_loss /= n_values_o
    reference_activation_loss /= n_values_a
    reference_weight_loss /= n_values_w

    return {
        'reference_activation_loss': reference_activation_loss,
        'reference_weight_loss': reference_weight_loss,
        'reference_output_loss': reference_output_loss,
    }


@torch.no_grad()
def evaluate_module_loss(module, w, a, quantizer, dtype=torch.float32, batch_size=64, prefix='initial_'):
    """Errors of the quantization in the basis of the transform `module`."""
    assert w.dtype == dtype, f"Weights must be in {dtype} for evaluation."
    assert a.dtype == dtype, f"Activations must be in {dtype} for evaluation."
    assert module.dtype == dtype, f"Module must be in {dtype} for evaluation."

    n_samples = a.shape[0]

    n_values_a = a.numel()
    n_values_w = w.numel()

    o = a @ w.t()
    n_values_o = o.numel()

    wt = module(w, inverse=True)
    q_wt = quantizer(wt)

    initial_weight_loss = (wt - q_wt).pow(2).sum().item() / n_values_w

    initial_activation_loss = 0.0
    initial_output_loss = 0.0
    fp_output_loss = 0.0
    relative_output_loss = 0.0

    for start in range(0, n_samples, batch_size):
        idx = slice(start, min(start + batch_size, n_samples))
        a_batch = a[idx]
        o_batch = o[idx]

        at_batch = module(a_batch, inverse=False)
        ot_batch = at_batch @ wt.t()

        q_at_batch = quantizer(at_batch)

        initial_activation_loss += (at_batch - q_at_batch).pow(2).sum().item()
        fp_output_loss += (o_batch - ot_batch).pow(2).sum().item()
        q_ot_batch = q_at_batch @ q_wt.t()

        relative_output_loss += (ot_batch - q_ot_batch).pow(2).sum().item()
        initial_output_loss += (o_batch - q_ot_batch).pow(2).sum().item()

    initial_activation_loss /= n_values_a
    initial_output_loss /= n_values_o
    fp_output_loss /= n_values_o
    relative_output_loss /= n_values_o

    return {
        f'{prefix}activation_loss': initial_activation_loss,
        f'{prefix}weight_loss': initial_weight_loss,
        f'{prefix}output_loss': initial_output_loss,
        f'{prefix}fp_output_loss': fp_output_loss,
        f'{prefix}relative_output_loss': relative_output_loss,
    }
