# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Builder script of the MJHQ-30K evaluation prompts (and of its images, the FID reference)."""

import json
import os

import datasets

from ._common import PromptsConfig, fixed_subset, name_seed, prompt_features

_REPOSITORY = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main"
_CITATION = 'Li et al., "Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation", arXiv:2402.17245, 2024.'


class Mjhq30k(datasets.GeneratorBasedBuilder):
    """30,000 prompts in 10 categories, each with a high-quality Midjourney image."""

    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIG_CLASS = PromptsConfig
    BUILDER_CONFIGS = [PromptsConfig(name="mjhq", version=VERSION, description="MJHQ-30K prompts")]
    DEFAULT_CONFIG_NAME = "mjhq"

    def _info(self):
        return datasets.DatasetInfo(
            description="MJHQ-30K: aesthetic-quality benchmark of Playground v2.5 (FID on high-quality images).",
            features=prompt_features(),
            homepage="https://huggingface.co/datasets/playgroundai/MJHQ-30K",
            license="Playground v2.5 Community License "
                    "(https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md)",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        paths = {"meta_path": dl_manager.download(f"{_REPOSITORY}/meta_data.json"), "image_root": None}
        if self.config.return_gt:
            paths["image_root"] = dl_manager.download_and_extract(f"{_REPOSITORY}/mjhq30k_imgs.zip")
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=paths)]

    def _generate_examples(self, meta_path: str, image_root: str | None):
        with open(meta_path, "r") as file:
            metadata = json.load(file)
        for index, name in enumerate(fixed_subset(list(metadata), self.config.max_dataset_size)):
            category = metadata[name]["category"]
            yield index, {
                "filename": f"{category}_{name}.png",
                "prompt": metadata[name]["prompt"],
                "seed": name_seed(name) + self.config.seed_offset,
                "image": os.path.join(image_root, category, f"{name}.jpg") if image_root else None,
            }
