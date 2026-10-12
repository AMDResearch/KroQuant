# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Prompt datasets. Each one is a Hugging Face `datasets` builder script of this folder; a batch
is a dict of lists with `filename`, `prompt`, `seed` (and `image` when requested)."""

from pathlib import Path
from typing import Iterator

import datasets
from loguru import logger

BUILDER_SCRIPTS = {"mjhq": "mjhq30k.py", "sdci": "sdci.py", "coco_captions_2024": "coco_captions.py"}


def load_prompts(name: str, num_prompts: int, batch_size: int, seed_offset: int = 0,
                 with_images: bool = False) -> tuple[Iterator[dict], int, int]:
    """(batches, number of prompts, number of batches) of the dataset `name`.

    `num_prompts > 0` selects a fixed pseudo-random subset, the same in every run (the subsets
    are nested). `seed_offset` shifts the noise seeds of MJHQ; 0 reproduces the paper.
    """
    if name not in BUILDER_SCRIPTS:
        raise ValueError(f"Unknown prompt dataset {name!r}; available: {', '.join(BUILDER_SCRIPTS)}.")
    if seed_offset and name != "mjhq":
        raise ValueError(f"seed_offset={seed_offset} is only implemented for mjhq.")
    builder_options = {"max_dataset_size": num_prompts, "return_gt": with_images}
    if name == "mjhq":
        builder_options["seed_offset"] = seed_offset
    script = Path(__file__).parent / BUILDER_SCRIPTS[name]
    logger.info(f"Reading the {name} prompts with {script.name}")
    table = datasets.load_dataset(str(script), name=name, split="train", trust_remote_code=True, **builder_options)
    if not isinstance(table, datasets.Dataset):
        raise TypeError(f"{script.name} produced a {type(table).__name__}, not a datasets.Dataset.")
    per_batch = max(1, batch_size)
    return table.iter(batch_size=per_batch), len(table), -(-len(table) // per_batch)
