# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Image metrics, updated batch by batch.

* LPIPS and PSNR compare each quantized image with the full-precision image of the same prompt.
* ImageReward scores both images against their prompt.
* FID compares the quantized images with the dataset images (clean-fid features, "clean" mode).
"""

import os
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio


class LPIPS:
    columns = ("lpips",)

    def __init__(self, device):
        self._metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    def update(self, batch: dict) -> None:
        self._metric.update(batch["reference"], batch["quantized"])

    def result(self) -> dict:
        return {"lpips": self._metric.compute().cpu().item()}


class PSNR:
    columns = ("psnr",)

    def __init__(self, device):
        self._metric = PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)).to(device)

    def update(self, batch: dict) -> None:
        self._metric.update(batch["reference"], batch["quantized"])

    def result(self) -> dict:
        return {"psnr": self._metric.compute().cpu().item()}


def _image_reward_cache() -> str | None:
    """A writable folder for the ImageReward weights: $IMAGEREWARD_CACHE, $XDG_CACHE_HOME/ImageReward,
    ~/.cache/ImageReward or ./.cache/ImageReward (None: the library default)."""
    candidates = [os.environ.get("IMAGEREWARD_CACHE")]
    if os.environ.get("XDG_CACHE_HOME"):
        candidates.append(os.path.join(os.environ["XDG_CACHE_HOME"], "ImageReward"))
    home = os.path.expanduser("~")
    if home and home != "~":
        candidates.append(os.path.join(home, ".cache", "ImageReward"))
    candidates.append(os.path.join(os.getcwd(), ".cache", "ImageReward"))
    for folder in filter(None, candidates):
        try:
            os.makedirs(folder, exist_ok=True)
            probe = Path(folder) / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return folder
        except OSError:
            continue
    return None


class ImageReward:
    columns = ("image_reward", "image_reward_reference")

    def __init__(self, device):
        import ImageReward as image_reward

        self._model = image_reward.load("ImageReward-v1.0", download_root=_image_reward_cache())
        self._quantized: list[float] = []
        self._reference: list[float] = []

    def update(self, batch: dict) -> None:
        for prompt, reference, quantized in zip(batch["prompt"], batch["reference_path"], batch["quantized_path"]):
            self._quantized.append(self._model.score(prompt, quantized))
            self._reference.append(self._model.score(prompt, reference))

    def result(self) -> dict:
        def mean(scores):
            return np.mean(scores).item() if scores else 0.0

        return {"image_reward": mean(self._quantized), "image_reward_reference": mean(self._reference)}


class FID:
    columns = ("fid",)
    needs_dataset_images = True

    def __init__(self, device):
        from cleanfid import fid as clean_fid

        self._clean_fid = clean_fid
        self._device = torch.device(device)
        self._extractor = clean_fid.build_feature_extractor("clean", self._device)
        self._fake: list[np.ndarray] = []
        self._real: list[np.ndarray] = []

    def update(self, batch: dict) -> None:
        for key, store in (("quantized_299", self._fake), ("dataset_299", self._real)):
            images = batch[key]
            if images.ndim != 4 or tuple(images.shape[1:]) != (3, 299, 299):
                raise ValueError(f"FID expects (n, 3, 299, 299) batches, got {tuple(images.shape)}.")
            store.append(self._clean_fid.get_batch_features(images, self._extractor, self._device))

    def result(self) -> dict:
        def statistics(chunks):
            features = np.concatenate(chunks, axis=0)
            return np.mean(features, axis=0), np.cov(features, rowvar=False)

        mu_real, sigma_real = statistics(self._real)
        mu_fake, sigma_fake = statistics(self._fake)
        distance = float(self._clean_fid.frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake))
        return {"fid": torch.tensor([distance], dtype=torch.float32).item()}


METRICS = {"lpips": LPIPS, "psnr": PSNR, "image_reward": ImageReward, "fid": FID}
