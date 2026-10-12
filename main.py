# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""KroQuant command line: one step per run.

    python main.py experiment=kronecker step=kronecker_init
    python main.py experiment=kronecker step=kronecker
    python main.py experiment=kronecker step=lowrank
    python main.py experiment=kronecker step=generate
    python main.py experiment=kronecker step=evaluate

Any setting can be overridden on the command line, e.g. `data=sdci` or `data.num_prompts=1000`.
"""

from kroquant.runtime import quiet_libraries

quiet_libraries()

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from kroquant import steps  # noqa: E402
from kroquant.runtime import calibration_device, seed_everything, start_logging  # noqa: E402


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log_dir = HydraConfig.get().runtime.output_dir
    start_logging(log_dir)
    calibration_device()
    seed_everything(cfg.seed)
    steps.run(cfg, log_dir)


if __name__ == "__main__":
    main()
