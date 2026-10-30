# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Scores the quantized images against the full-precision references (and the dataset images)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cleanfid.resize import build_resizer
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ..data import load_prompts
from .metrics import METRICS


class ImagePairs(Dataset):
    """For every prompt: the full-precision and quantized images (matched by file name), and, for
    FID, both resized to 299x299 by clean-fid's "clean" resizer (the quantized image and the
    dataset image)."""

    def __init__(self, names: list[str], prompts: list[str], reference_dir: Path, quantized_dir: Path,
                 dataset_images: dict | None):
        self.names, self.prompts = names, prompts
        self.reference_dir, self.quantized_dir = Path(reference_dir), Path(quantized_dir)
        self.dataset_images = dataset_images
        self.to_tensor = transforms.ToTensor()
        self.resize_299 = build_resizer("clean")

    def __len__(self) -> int:
        return len(self.names)

    def _fid_input(self, image: Image.Image) -> torch.Tensor:
        resized = self.resize_299(np.array(image))
        tensor = self.to_tensor(resized)
        return tensor * 255 if resized.dtype == "uint8" else tensor

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        reference_path, quantized_path = self.reference_dir / name, self.quantized_dir / name
        reference = Image.open(reference_path).convert("RGB")
        quantized = Image.open(quantized_path).convert("RGB")
        if reference.size != quantized.size:
            reference = reference.resize(quantized.size, Image.Resampling.BICUBIC)
        item = {
            "reference": self.to_tensor(reference),
            "quantized": self.to_tensor(quantized),
            "prompt": self.prompts[index],
            "reference_path": str(reference_path),
            "quantized_path": str(quantized_path),
        }
        if self.dataset_images is not None:
            item["quantized_299"] = self._fid_input(quantized)
            item["dataset_299"] = self._fid_input(self.dataset_images[name].convert("RGB"))
        return item


def evaluate(data, metric_names: list[str], reference_dir: Path, quantized_dir: Path, num_workers: int = 8,
             device: str = "cuda") -> dict:
    unknown = set(metric_names) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; available: {', '.join(METRICS)}.")
    with_images = any(getattr(METRICS[name], "needs_dataset_images", False) for name in metric_names)
    batches, _, n_batches = load_prompts(data.name, data.num_prompts, batch_size=32, with_images=with_images)
    names, prompts, dataset_images = [], [], {} if with_images else None
    for batch in tqdm(batches, total=n_batches, desc="Reading the prompts", leave=False):
        names += batch["filename"]
        prompts += batch["prompt"]
        if with_images:
            dataset_images.update(zip(batch["filename"], batch["image"]))

    available = {path.name for path in Path(reference_dir).glob("*.png")} & {path.name for path in Path(quantized_dir).glob("*.png")}
    missing = set(names) - available
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) missing in {reference_dir} or {quantized_dir}: run step=generate first.")

    metrics = [METRICS[name](device) for name in metric_names]
    loader = DataLoader(ImagePairs(names, prompts, reference_dir, quantized_dir, dataset_images),
                        batch_size=data.eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    for batch in tqdm(loader, desc="Scoring"):
        batch["reference"] = batch["reference"].to(device)
        batch["quantized"] = batch["quantized"].to(device)
        for metric in metrics:
            metric.update(batch)
    scores = {}
    for metric in metrics:
        scores.update(metric.result())
    torch.cuda.empty_cache()
    return scores


def write_scores(scores: dict, *paths: Path) -> None:
    """`<path>.csv` and `<path>.md` for every path."""
    table = pd.DataFrame([scores])
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(f"{path}.csv", index=False)
        Path(f"{path}.md").write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")
    logger.info("Scores: " + ", ".join(f"{key}={value:.4g}" for key, value in scores.items()))
