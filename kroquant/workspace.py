# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Where every step reads and writes, and the settings guard of the output folders.

    <workdir>/calibration_inputs.pt           recorded calibration inputs (shared by experiments)
    <workdir>/reference/<data>/*.png          full-precision images (shared by experiments)
    <workdir>/<name>/<stage>/                 results of a calibration stage
    <workdir>/<name>/images/<data>/*.png      images of the quantized model
    <workdir>/<name>/metrics/<data>.{csv,md}  scores

Each of these folders (and the calibration inputs) has a `settings.yaml` with the settings its
content was produced with, including those of the earlier stages. Reusing a folder with other
settings is refused instead of silently mixing results.
"""

from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

STAGES = ("kronecker_init", "kronecker", "lowrank")


class SettingsMismatch(RuntimeError):
    pass


def _plain(node) -> dict:
    return OmegaConf.to_container(node, resolve=True) if isinstance(node, DictConfig) else dict(node)


def model_settings(cfg: DictConfig) -> dict:
    model = _plain(cfg.model)
    return {key: model[key] for key in ("name", "repo", "pipeline", "torch_dtype", "load_kwargs", "steps", "guidance")
            if key in model}


def calibration_settings(cfg: DictConfig) -> dict:
    return {"model": model_settings(cfg), "calibration": _plain(cfg.calibration)}


def stage_settings(cfg: DictConfig, stage: str) -> dict:
    """Everything the results of `stage` depend on: the model, the format, the calibration inputs,
    and the settings of this stage and of the earlier ones."""
    settings = calibration_settings(cfg)
    settings["format"] = cfg.format
    settings["transform"] = _plain(cfg.transform)
    for name in STAGES[: STAGES.index(stage) + 1]:
        settings[name] = _plain(cfg[name])
    return settings


def data_settings(cfg: DictConfig) -> dict:
    data = _plain(cfg.data)
    return {key: data[key] for key in ("name", "num_prompts", "batch_size", "seed_offset")}


def data_tag(cfg: DictConfig) -> str:
    return f"{cfg.data.name}_seed{cfg.data.seed_offset}" if cfg.data.seed_offset else cfg.data.name


class Workspace:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root = Path(cfg.workdir)
        self.experiment = self.root / cfg.name

    @property
    def calibration_inputs(self) -> Path:
        return self.root / "calibration_inputs.pt"

    def stage(self, name: str) -> Path:
        return self.experiment / name

    @property
    def reference_images(self) -> Path:
        return self.root / "reference" / data_tag(self.cfg)

    @property
    def quantized_images(self) -> Path:
        return self.experiment / "images" / data_tag(self.cfg)

    @property
    def scores(self) -> Path:
        return self.experiment / "metrics" / data_tag(self.cfg)

    @staticmethod
    def claim(settings_file: Path, settings: dict) -> None:
        """Record `settings` for the outputs next to `settings_file`, or check that they match."""
        settings = yaml.safe_load(yaml.safe_dump(settings))
        if settings_file.exists():
            stored = yaml.safe_load(settings_file.read_text(encoding="utf-8"))
            if stored != settings:
                changed = sorted(key for key in set(stored) | set(settings) if stored.get(key) != settings.get(key))
                raise SettingsMismatch(
                    f"{settings_file.parent} holds outputs produced with other settings (differences in: "
                    f"{', '.join(changed)}). Choose another `name=` (or `workdir=`), or delete that folder."
                )
            return
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    def claim_calibration_inputs(self) -> None:
        self.claim(self.calibration_inputs.with_suffix(".yaml"), calibration_settings(self.cfg))

    def claim_stage(self, name: str) -> None:
        self.claim(self.stage(name) / "settings.yaml", stage_settings(self.cfg, name))

    def require_stage(self, name: str) -> None:
        """Installing a stage needs its results, produced with the current settings."""
        settings_file = self.stage(name) / "settings.yaml"
        if not settings_file.exists():
            raise FileNotFoundError(f"Stage {name} has not been calibrated in {self.stage(name)}: run step={name} first.")
        self.claim(settings_file, stage_settings(self.cfg, name))

    def claim_reference_images(self) -> None:
        self.claim(self.reference_images / "settings.yaml",
                   {"model": model_settings(self.cfg), "image_size": [self.cfg.model.width, self.cfg.model.height],
                    "data": data_settings(self.cfg), "seed": self.cfg.seed})

    def claim_quantized_images(self) -> None:
        settings = stage_settings(self.cfg, STAGES[-1])
        settings.update({"quantized_layers": list(self.cfg.model.layers),
                         "image_size": [self.cfg.model.width, self.cfg.model.height],
                         "data": data_settings(self.cfg), "seed": self.cfg.seed})
        self.claim(self.quantized_images / "settings.yaml", settings)
