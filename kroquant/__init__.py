# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""KroQuant: post-training quantization of diffusion transformers with Kronecker-structured
block transforms, followed by a quantized low-rank branch.

Entry point: `main.py` (one Hydra run per step, see `kroquant.steps`).
"""
