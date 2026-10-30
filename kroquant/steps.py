# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""The five steps of KroQuant, one per run of `main.py` (`step=<name>`):

    kronecker_init  Kronecker transforms at their randomized Hadamard initialization (0 epochs)
    kronecker       Kronecker transforms learned on calibration inputs
    lowrank         quantized low-rank branch (LoRaQ)
    generate        full-precision images, then images of the quantized model
    evaluate        metrics of the quantized images

A calibration step installs the stages before it; generation installs all three, then quantizes
the inputs and weights of every layer of `model.layers`.
"""

import torch
from loguru import logger
from omegaconf import DictConfig

from .calibration import BlockInputs
from .calibration.layerwise import LayerwiseStage
from .data import load_prompts
from .hooks import HookSet
from .images import Sampling, evaluate, render, write_scores
from .kronecker import KroneckerStage, TransformSpec
from .lowrank import LowRankStage
from .models import Pipeline, load_diffusers, matches
from .quant import Quantizer, quantize_inputs, quantize_weight_
from .runtime import calibration_device, visible_gpus
from .workspace import STAGES, Workspace


def build_stage(cfg: DictConfig, name: str, workspace: Workspace, device: torch.device) -> LayerwiseStage:
    if name == "lowrank":
        section = cfg.lowrank
        fit = {"lr": section.lr, "epochs": section.epochs,
               "rotation_lr": section.rotation_lr, "rotation_epochs": section.rotation_epochs}
        return LowRankStage(name, list(section.layers), workspace.stage(name), device, fmt=cfg.format,
                            factor_format=section.factor_format, rank=section.rank, svd_rank=section.svd_rank, fit=fit)
    section, transform = cfg[name], cfg.transform
    return KroneckerStage(name, list(section.layers), workspace.stage(name), device, fmt=cfg.format,
                          spec=TransformSpec(N=transform.factors, random_init=transform.random_signs),
                          lr=transform.lr, epochs=section.epochs, batch_size=transform.batch_size)


def install(stages: list[LayerwiseStage], pipe: Pipeline, workspace: Workspace) -> HookSet:
    """Install the calibrated `stages` on `pipe`, in order (their results must exist)."""
    hooks = HookSet()
    try:
        for stage in stages:
            workspace.require_stage(stage.name)
            logger.info(f"Installing {stage.name} from {stage.directory}")
            hooks.add(stage.attach(pipe))
    except BaseException:
        hooks.remove()
        raise
    return hooks


def calibrate(cfg: DictConfig, name: str) -> None:
    workspace = Workspace(cfg)
    device = calibration_device()
    stages = [build_stage(cfg, stage, workspace, device) for stage in STAGES[: STAGES.index(name) + 1]]
    *earlier, current = stages
    workspace.claim_calibration_inputs()
    workspace.claim_stage(name)

    pipe = Pipeline.load(cfg.model).to(device)
    calibration = cfg.calibration
    prompts, _, n_batches = load_prompts(calibration.prompts, calibration.num_prompts, calibration.batch_size)
    with install(earlier, pipe, workspace):
        current.calibrate(
            pipe, BlockInputs(workspace.calibration_inputs, device), prompts, n_batches,
            num_samples=calibration.num_samples, sample_seed=calibration.sample_seed,
            steps=cfg.model.steps, guidance=cfg.model.guidance,
        )


def _release(pipes: list) -> None:
    pipes.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    sampling = Sampling.of(cfg.model, cfg.seed)
    gpus = visible_gpus()
    data = cfg.data

    workspace.claim_reference_images()
    batches, total, n_batches = load_prompts(data.name, data.num_prompts, data.batch_size, data.seed_offset)
    reference = workspace.reference_images
    if len(list(reference.glob("*.png"))) < total:
        pipes = []
        for gpu in gpus:
            pipe = load_diffusers(cfg.model).to(gpu)
            pipe.set_progress_bar_config(disable=True)
            pipes.append(pipe)
        render(pipes, batches, n_batches, reference, sampling)
        _release(pipes)

    workspace.claim_quantized_images()
    batches, total, n_batches = load_prompts(data.name, data.num_prompts, data.batch_size, data.seed_offset)
    weights = Quantizer(cfg.format)
    pipes, hooks = [], HookSet()
    try:
        for gpu in gpus:
            pipe = Pipeline.load(cfg.model).to(gpu)
            pipe.silence_progress_bars()
            pipes.append(pipe)
        stages = [build_stage(cfg, stage, workspace, gpus[0]) for stage in STAGES]
        for pipe in pipes:
            hooks.add(install(stages, pipe, workspace))
        for pipe in pipes:
            activations = Quantizer(cfg.format)
            for layer in pipe.layers():
                if matches(layer.name, cfg.model.layers):
                    hooks.add(quantize_inputs(layer.module, activations))
                    quantize_weight_(layer.module, weights)
        render(pipes, batches, n_batches, workspace.quantized_images, sampling)
    finally:
        hooks.remove()
        _release(pipes)


def score(cfg: DictConfig, log_dir: str) -> None:
    workspace = Workspace(cfg)
    scores = evaluate(cfg.data, list(cfg.evaluation.metrics), workspace.reference_images, workspace.quantized_images)
    write_scores(scores, workspace.scores, f"{log_dir}/scores")


def run(cfg: DictConfig, log_dir: str) -> None:
    if cfg.step in STAGES:
        calibrate(cfg, cfg.step)
    elif cfg.step == "generate":
        generate(cfg)
    elif cfg.step == "evaluate":
        score(cfg, log_dir)
    else:
        raise ValueError(f"Unknown step {cfg.step!r}; one of {', '.join((*STAGES, 'generate', 'evaluate'))}.")
