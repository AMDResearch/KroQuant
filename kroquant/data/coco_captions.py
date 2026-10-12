# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Builder script of the calibration prompts: 1024 MS-COCO captions, the calibration set of
SVDQuant / DeepCompressor (stored in `coco_captions_2024.json`, next to this script)."""

import json

import datasets

from ._common import PromptsConfig, fixed_subset, prompt_features, text_seed

_CAPTIONS = "coco_captions_2024.json"
_CITATION = 'Li, Lin, Zhang et al., "SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models", ICLR 2025.'


class CocoCaptions2024(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIG_CLASS = PromptsConfig
    BUILDER_CONFIGS = [PromptsConfig(name="coco_captions_2024", version=VERSION, description="COCO calibration captions",
                                     max_dataset_size=1000, return_gt=False)]
    DEFAULT_CONFIG_NAME = "coco_captions_2024"

    def _info(self):
        return datasets.DatasetInfo(
            description="Captions of MS-COCO, selected for calibration in DeepCompressor "
                        "(https://github.com/nunchaku-tech/deepcompressor/blob/main/examples/diffusion/prompts/qdiff.yaml).",
            features=prompt_features(with_image=False),
            homepage="https://cocodataset.org",
            license="COCO annotations: Creative Commons Attribution 4.0 (CC BY 4.0)",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"captions_path": dl_manager.download(_CAPTIONS)})]

    def _generate_examples(self, captions_path: str):
        with open(captions_path, "r", encoding="utf-8") as file:
            captions = list(json.load(file).items())
        if self.config.max_dataset_size > 0:
            captions = sorted(fixed_subset(captions, self.config.max_dataset_size), key=lambda pair: pair[0])
        for index, (uid, caption) in enumerate(captions):
            yield index, {"filename": uid, "prompt": caption, "seed": text_seed(caption)}
