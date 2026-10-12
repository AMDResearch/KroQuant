# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Shared by the dataset builder scripts of this folder.

Hugging Face `datasets` copies this file next to each script that imports it relatively.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import List, TypeVar

import datasets

Item = TypeVar("Item")


def name_seed(name: str) -> int:
    """Noise seed of an image: the first four bytes of SHA-256(name), as a signed 32-bit integer.

    Every method generates an image from the same noise, so comparisons are paired.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=True)


def text_seed(text: str) -> int:
    """Noise seed derived from a prompt: polynomial hash in base 31, modulo 1_000_000_007."""
    seed = 0
    for character in text:
        seed = (seed * 31 + ord(character)) % 1_000_000_007
    return seed


def fixed_subset(items: List[Item], size: int) -> List[Item]:
    """`size` items in a fixed pseudo-random order (seed 0); every item, in order, if size <= 0.

    The subsets are nested: the first n items of a larger subset form the subset of size n.
    """
    if size <= 0:
        return items
    shuffled = list(items)
    random.Random(0).shuffle(shuffled)
    return shuffled[:size]


@dataclass
class PromptsConfig(datasets.BuilderConfig):
    """`max_dataset_size` prompts (all if <= 0), with the reference images if `return_gt`;
    `seed_offset` shifts every noise seed (MJHQ only)."""

    max_dataset_size: int = -1
    return_gt: bool = False
    seed_offset: int = 0


def prompt_features(with_image: bool = True) -> datasets.Features:
    columns = {"filename": datasets.Value("string"), "prompt": datasets.Value("string"), "seed": datasets.Value("int32")}
    if with_image:
        columns["image"] = datasets.Image()
    return datasets.Features(columns)
