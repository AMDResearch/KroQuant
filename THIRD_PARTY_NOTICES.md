# Third-party notices

KroQuant's own code is under the MIT license (`LICENSE`). Everything listed here comes from
elsewhere and stays under its own license.

## Included in this repository

| Component | Where | License |
|---|---|---|
| TensorCast, https://github.com/ROCm/tensorcast, commit `56c51c955f07d71eb1b42157733d173f4b20cbc0` | git submodule `third-parties/tensorcast`, unmodified; `kroquant/quant/tensorcast.py` changes two behaviours when it is imported | MIT, Copyright (c) 2024 AMD ROCm Software |
| 1024 MS-COCO captions, in the selection of DeepCompressor (`examples/diffusion/prompts/qdiff.yaml`, https://github.com/nunchaku-tech/deepcompressor, Apache-2.0) | `kroquant/data/coco_captions_2024.json`, calibration prompts | COCO annotations: CC BY 4.0, https://cocodataset.org |

The Python packages of `requirements.txt` are installed separately and come with their own licenses.

## Downloaded on first use

| Asset | Used for | License |
|---|---|---|
| PixArt-Σ, `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | quantized model | CreativeML Open RAIL++-M |
| SANA 1600M, `Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632` (BF16 conversion of `Efficient-Large-Model/Sana_1600M_1024px_diffusers`) | `model=sana` | Apache-2.0 (original checkpoint) |
| FLUX.1 [schnell], `black-forest-labs/FLUX.1-schnell` | `model=flux_schnell` | Apache-2.0 |
| FLUX.1 [dev], `black-forest-labs/FLUX.1-dev` | `model=flux_dev` | FLUX.1 [dev] Non-Commercial License |
| MJHQ-30K, `playgroundai/MJHQ-30K` | evaluation prompts, FID reference images | Playground v2.5 Community License |
| sDCI, `mit-han-lab/svdquant-datasets` (from DCI, SA-1B images) | evaluation prompts, FID reference images | CC BY-NC 4.0 |
| ImageReward, `THUDM/ImageReward` | ImageReward metric | Apache-2.0 |
| LPIPS weights, through torchmetrics | LPIPS metric | BSD-2-Clause |
| Inception weights, through clean-fid | FID | see https://github.com/GaParmar/clean-fid |

