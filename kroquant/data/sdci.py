# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Builder script of the sDCI evaluation prompts (summarized Densely Captioned Images), as
distributed with SVDQuant, and of its images (the FID reference)."""

import os

import datasets
import yaml

from ._common import PromptsConfig, fixed_subset, name_seed, prompt_features

_REPOSITORY = "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main"
_CITATION = 'Urbanek et al., "A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions", CVPR 2024.'


class SummarizedDci(datasets.GeneratorBasedBuilder):
    """Short captions of the Densely Captioned Images (SA-1B images)."""

    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIG_CLASS = PromptsConfig
    BUILDER_CONFIGS = [PromptsConfig(name="sdci", version=VERSION, description="sDCI prompts")]
    DEFAULT_CONFIG_NAME = "sdci"

    def _info(self):
        return datasets.DatasetInfo(
            description="DCI: 7805 SA-1B images, each with a complete description; sDCI keeps short summaries.",
            features=prompt_features(),
            homepage="https://github.com/facebookresearch/DCI",
            license="Attribution-NonCommercial 4.0 International (https://github.com/facebookresearch/DCI/blob/main/LICENSE)",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        paths = {"meta_path": dl_manager.download(f"{_REPOSITORY}/sDCI.yaml"), "image_root": None}
        if self.config.return_gt:
            paths["image_root"] = dl_manager.download_and_extract(f"{_REPOSITORY}/sDCI.gz")
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=paths)]

    def _generate_examples(self, meta_path: str, image_root: str | None):
        with open(meta_path, "r") as file:
            captions = yaml.safe_load(file)
        for index, name in enumerate(fixed_subset(list(captions), self.config.max_dataset_size)):
            yield index, {
                "filename": f"{name}.png",
                "prompt": captions[name],
                "seed": name_seed(name),
                "image": os.path.join(image_root, f"{name}.jpg") if image_root else None,
            }
