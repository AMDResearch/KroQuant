# KroQuant

**KroQuant: Kronecker-Structured Block Transforms for Efficient Post-Training Quantization of Diffusion Transformers**
Yann Bouquet, Alireza Khodamoradi, Kristof Denolf, Mathieu Salzmann (EPFL, AMD). NeurIPS 2026.
[Paper](https://arxiv.org/abs/2607.21446) 

KroQuant quantizes every linear layer of a diffusion transformer to 4 bits, weights and
activations, without retraining. Before quantizing a layer `y = x W^T`, it rotates its input
features with a learned block-diagonal matrix `M`, applied as `(x M)(W M^-T)^T`. The output is
unchanged in full precision, but the outliers spread over the block and both operands round
with less error. Each 32 × 32 block of `M` is a Kronecker product of five 2 × 2 factors: 15
parameters per block instead of 1024. A quantized low-rank branch then absorbs part of the
remaining weight error.

This repository reproduces the PixArt-Σ experiment of the paper:

| | |
|---|---|
| model | PixArt-Σ 1024 px (configurations for SANA and FLUX.1 are included) |
| quantization | W4A4 MXFP4 (`mxfp4e2`: e2m1 values sharing one e8m0 scale per 32) on all 224 linear layers of the transformer blocks |
| benchmarks | MJHQ-30K and sDCI, 5000 prompts each: FID, ImageReward, LPIPS and PSNR against the 16-bit model |

## How it works

Calibration runs in three stages. Each stage keeps what the previous ones installed.

1. **`kronecker_init`**: every 2 × 2 factor starts at `H_2 / sqrt(2)`, so each block starts as a
   Hadamard matrix with random signs.
2. **`kronecker`**: the factors are trained on calibration activations.
   The loss is the output error of the quantized layer, regularized by the weight and
   activation errors, whose coefficients are set from their initial magnitudes.
3. **`lowrank`**: every layer gets a rank-128 MXFP4 branch `L R`, and the residual `W - L R`
   is quantized. `L` and `R` start from the truncated SVD of the transformed weight. They are
   fitted so that the residual quantizes well, then rotated (`L r`, `rᵀ R`) by an orthogonal
   `r` learned on the Stiefel manifold, so that the factors quantize well too. This stage is
   the method of LoRaQ (Bouquet et al., NeurIPS 2026).

## Setup

You need Python 3.12 and an NVIDIA GPU with CUDA 12.8 (tested on one A100 80 GB).

```bash
git clone --recurse-submodules <repository url> kroquant && cd kroquant
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Quantization goes through [TensorCast](https://github.com/ROCm/tensorcast), which is vendored as
the git submodule `third-parties/tensorcast` (commit `56c51c9`); it is not installed with pip. If that
folder is empty:

```bash
git submodule update --init third-parties/tensorcast
# without git: git clone https://github.com/ROCm/tensorcast.git third-parties/tensorcast && git -C third-parties/tensorcast checkout 56c51c9
```

Models and datasets are downloaded from the Hugging Face Hub when first needed. The MJHQ-30K
images, used only by FID, take a few GB; `HF_HOME` sets the cache location.

## Running the experiment

Each step is one run of `main.py`. A calibration step needs the stages before it.

| step | command | time on one A100 |
|---|---|---|
| 1 | `python main.py experiment=kronecker step=kronecker_init` | 2 min |
| 2 | `python main.py experiment=kronecker step=kronecker` | 2 h |
| 3 | `python main.py experiment=kronecker step=lowrank` | 15 min |
| 4 | `python main.py experiment=kronecker step=generate` | ~4 h per 5000 images (16-bit, then quantized) |
| 5 | `python main.py experiment=kronecker step=evaluate` | minutes |

For sDCI, add `data=sdci` to steps 4 and 5. Any setting of `conf/` can be overridden in the same
way, e.g. `data.num_prompts=1000`.

## Paper numbers

PixArt-Σ, W4A4 MXFP4, 5000 images per benchmark:

| method | MJHQ FID ↓ | MJHQ IR ↑ | MJHQ LPIPS ↓ | MJHQ PSNR ↑ | sDCI FID ↓ | sDCI IR ↑ | sDCI LPIPS ↓ | sDCI PSNR ↑ |
|---|---|---|---|---|---|---|---|---|
| 16-bit | 16.6 | 0.948 | – | – | 24.8 | 0.966 | – | – |
| KroQuant | 16.9 | 0.989 | 0.347 | 16.5 | 21.9 | 1.01 | 0.378 | 15.2 |

Other GPUs and library versions give slightly different values.

## Where the outputs go

```text
outputs/pixart_sigma/                          workdir (one per model)
├── calibration_inputs.pt                      recorded calibration inputs, shared by experiments
├── reference/mjhq/*.png                       16-bit images, shared by experiments
└── kronecker/                                 one folder per experiment name
    ├── kronecker_init/transforms.pt           per-layer transforms; plots/: training curves
    ├── kronecker/transforms.pt
    ├── lowrank/factors.pt, svd.pt             per-layer L, R (+ diagnostics); plots/: errors before/after
    ├── images/mjhq/*.png                      images of the quantized model
    ├── metrics/mjhq.csv, mjhq.md
    └── logs/<step>/<date>_<time>/             log and resolved configuration of every run
```

## Reproducibility

- Calibration uses a single GPU (the first visible one: choose it with `CUDA_VISIBLE_DEVICES`).
  Image generation uses every visible GPU.
- Every image has its own noise seed, derived from its file name, so all methods see the same
  noise. Images still depend on how prompts are batched: keep `data.batch_size` (8).
  `data.seed_offset=<n>` draws other seeds (MJHQ only).

## Other models

`conf/model/` also describes SANA 1600M (`sana`), FLUX.1 [schnell] (`flux_schnell`) and FLUX.1
[dev] (`flux_dev`), whose block structures (joint and single-stream blocks, convolutional MLPs)
the code supports. To quantize one, add an experiment file next to
`conf/experiment/kronecker.yaml`. 

## Cite

```bibtex
@inproceedings{bouquet2026kroquant,
  title     = {KroQuant: Kronecker-Structured Block Transforms for Efficient Post-Training Quantization of Diffusion Transformers},
  author    = {Bouquet, Yann and Khodamoradi, Alireza and Denolf, Kristof and Salzmann, Mathieu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

The code is released under the MIT license, Copyright (c) 2026 Advanced Micro Devices, Inc.
(`LICENSE`), and was written by Yann Bouquet. TensorCast, the models, the datasets and the
metric weights keep their own licenses (`THIRD_PARTY_NOTICES.md`).
