# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Image generation, one pipeline per visible GPU.

Every image has its own CPU generator, seeded from its name (see `kroquant.data`), and the global
seeds are reset before each batch. Images also depend on how prompts are batched
(https://github.com/huggingface/diffusers/issues/2659): keep the batch size when comparing runs.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import diffusers
import torch
from loguru import logger
from tqdm import tqdm


@dataclass(frozen=True)
class Sampling:
    """Arguments of the pipeline calls."""

    steps: int
    guidance: float
    width: int
    height: int
    seed: int = 0

    @classmethod
    def of(cls, model, seed: int) -> "Sampling":
        return cls(steps=model.steps, guidance=model.guidance, width=model.width, height=model.height, seed=seed)


def existing_images(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.png")} if directory.is_dir() else set()


@torch.inference_mode()
def render_batch(pipe: Callable, batch: dict, directory: Path, sampling: Sampling) -> None:
    """Generate the images of one batch and save those that do not exist yet."""
    torch.manual_seed(sampling.seed)
    diffusers.training_utils.set_seed(sampling.seed)
    generators = [torch.Generator().manual_seed(int(seed)) for seed in batch["seed"]]
    images = pipe(
        prompt=batch["prompt"],
        negative_prompt="",
        num_images_per_prompt=1,
        num_inference_steps=sampling.steps,
        guidance_scale=sampling.guidance,
        width=sampling.width,
        height=sampling.height,
        generator=generators,
    ).images
    for filename, image in zip(batch["filename"], images):
        path = directory / filename
        if not path.exists():
            image.save(path)


def render(pipes: list, batches: Iterable[dict], n_batches: int, directory: Path, sampling: Sampling) -> None:
    """Generate the missing images of `directory`: a batch is skipped when all its images exist.

    The batches go to the pipelines in rounds (one batch per pipeline); a failure in any of them
    stops the generation and is raised here.
    """
    directory.mkdir(parents=True, exist_ok=True)
    done = existing_images(directory)
    pending = (batch for batch in batches if not set(batch["filename"]) <= done)
    with ThreadPoolExecutor(max_workers=len(pipes)) as pool, tqdm(total=n_batches, desc=f"Images in {directory}") as bar:
        while True:
            round_ = [batch for _, batch in zip(pipes, pending)]
            if not round_:
                break
            jobs = [pool.submit(render_batch, pipe, batch, directory, sampling) for pipe, batch in zip(pipes, round_)]
            for job in jobs:
                job.result()
            bar.update(len(round_))
    logger.info(f"{len(existing_images(directory))} images in {directory}")
