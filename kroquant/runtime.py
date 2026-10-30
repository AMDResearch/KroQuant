# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Process-level setup shared by every step: quieter libraries, logging, CUDA device, seeds."""

import logging
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def quiet_libraries() -> None:
    """Silence the advisory messages of transformers / diffusers (call before importing them)."""
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for category in (FutureWarning, UserWarning):
        warnings.filterwarnings("ignore", category=category)
    for name in ("transformers", "diffusers"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _console(message) -> None:
    tqdm.write(str(message), end="", file=sys.stdout)


def start_logging(log_dir: str | Path) -> None:
    """INFO messages on the console (printed above the progress bars), everything in `kroquant.log`."""
    logger.remove()
    logger.add(_console, level="INFO", colorize=True)
    logger.add(Path(log_dir) / "kroquant.log", level="DEBUG", colorize=False)


def calibration_device() -> torch.device:
    """The GPU used by calibration: the first visible one (select it with CUDA_VISIBLE_DEVICES)."""
    if not torch.cuda.is_available():
        raise RuntimeError("KroQuant needs a CUDA GPU.")
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def visible_gpus() -> list[torch.device]:
    return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU). Called once per step, before any model is loaded."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
