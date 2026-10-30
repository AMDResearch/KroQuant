# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""The one entry point to TensorCast (https://github.com/ROCm/tensorcast) in this package.

TensorCast is vendored, unmodified, as the git submodule `third-parties/tensorcast` (tested at
`TESTED_COMMIT`); an installed `tcast` package is used as a fallback. Calibration needs two
behaviours that differ from upstream, so they are adjusted once, when this module is imported:

* `Cast.cast` is decorated with `torch.inference_mode` upstream. Tensors created in inference
  mode cannot be recorded by autograd, and the Kronecker and low-rank stages differentiate
  through quantized tensors, so the decorator is taken off.
* `Cast._get_scales` is given a detached input: the shared scales (and zero points) of a
  quantization block are constants of the grid, and no gradient flows through them.

A short check runs right after the adjustment.
"""

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType

import torch
from loguru import logger

TESTED_COMMIT = "56c51c955f07d71eb1b42157733d173f4b20cbc0"
VENDORED_DIR = Path(__file__).resolve().parents[2] / "third-parties" / "tensorcast"

_CAST_PARAMETERS = ["cls", "x", "dtype", "castmode", "roundmode", "scalemode", "compmode",
                    "scaledata", "better", "select", "noreshape"]
_SCALES_PARAMETERS = ["cls", "x", "dtype", "noreshape"]


def _find_tensorcast() -> ModuleType:
    if "tcast" in sys.modules:
        return sys.modules["tcast"]
    if (VENDORED_DIR / "tcast" / "__init__.py").is_file():
        sys.path.insert(0, str(VENDORED_DIR))
        try:
            return importlib.import_module("tcast")
        finally:
            sys.path.remove(str(VENDORED_DIR))
    try:
        module = importlib.import_module("tcast")
    except ImportError as error:
        raise ImportError(
            f"TensorCast is missing. Fetch it with `git submodule update --init third-parties/tensorcast` "
            f"or clone https://github.com/ROCm/tensorcast at {TESTED_COMMIT} into {VENDORED_DIR}."
        ) from error
    logger.warning(f"Using the TensorCast installed at {module.__file__}; KroQuant is tested with {TESTED_COMMIT}.")
    return module


def _require_parameters(function, expected: list[str]) -> None:
    found = list(inspect.signature(function).parameters)
    if found != expected:
        raise RuntimeError(f"Unsupported TensorCast version: {function.__qualname__}{found}, expected {expected}.")


def _without_inference_mode(function):
    """`function` itself if it is wrapped by a `torch.inference_mode` decorator, else unchanged."""
    if not hasattr(function, "__wrapped__"):
        return function
    for cell in function.__closure__ or ():
        content = cell.cell_contents
        if isinstance(content, torch.inference_mode) or isinstance(getattr(content, "__self__", None), torch.inference_mode):
            return function.__wrapped__
    return function


def _adjust(module: ModuleType) -> None:
    cast_class = module.Cast
    if getattr(cast_class, "_kroquant_adjusted", False):
        return

    cast = _without_inference_mode(cast_class.__dict__["cast"].__func__)
    _require_parameters(cast, _CAST_PARAMETERS)
    cast_class.cast = classmethod(cast)

    scales = cast_class.__dict__["_get_scales"].__func__
    _require_parameters(scales, _SCALES_PARAMETERS)

    def scales_of_detached(cls, x, dtype, noreshape=False):
        return scales(cls, x.detach(), dtype, noreshape)

    cast_class._get_scales = classmethod(scales_of_detached)
    cast_class._kroquant_adjusted = True


def _check(module: ModuleType) -> None:
    probe = torch.linspace(-4.0, 4.0, 64, requires_grad=True)
    mxfp4 = module.DataType("e2m1fnuz", "e8m0_t32", "mxfp4e2")
    if module.cast(probe, mxfp4).tensor.is_inference():
        raise RuntimeError("TensorCast adjustment failed: Cast.cast still runs in inference mode.")
    if module.Cast._get_scales(probe, mxfp4).scale.requires_grad:
        raise RuntimeError("TensorCast adjustment failed: the scales are not detached.")


def _checkout_commit() -> str | None:
    """Commit of the vendored checkout, read from its git metadata (None if unavailable)."""
    git = VENDORED_DIR / ".git"
    try:
        if git.is_file():
            git = (VENDORED_DIR / git.read_text().split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            head = (git / head.split(" ", 1)[1]).read_text().strip()
        return head
    except (OSError, IndexError):
        return None


tcast = _find_tensorcast()
_adjust(tcast)
_check(tcast)
if (commit := _checkout_commit()) not in (None, TESTED_COMMIT):
    logger.warning(f"third-parties/tensorcast is at {commit}; KroQuant is tested with {TESTED_COMMIT}.")

DataType = tcast.DataType

__all__ = ["tcast", "DataType", "TESTED_COMMIT"]
