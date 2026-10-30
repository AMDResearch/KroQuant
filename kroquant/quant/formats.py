# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Number formats that the configs can name.

All are OCP microscaling (MX) formats: blocks of 32 consecutive values along the last dimension
share one e8m0 (power-of-two) scale. The paper uses `mxfp4e2` (e2m1 elements).
"""

from .tensorcast import DataType

MX_FORMATS: dict[str, tuple[str, str]] = {
    "mxfp8e5": ("e5m2", "e8m0_t32"),
    "mxfp8e4": ("e4m3fn", "e8m0_t32"),
    "mxfp6e3": ("e3m2fnuz", "e8m0_t32"),
    "mxfp6e2": ("e2m3fnuz", "e8m0_t32"),
    "mxfp4e2": ("e2m1fnuz", "e8m0_t32"),
    "mxint8": ("int8", "e8m0_t32"),
    "mxint4": ("int4", "e8m0_t32"),
}


def number_format(name: str) -> DataType:
    """A new TensorCast `DataType` for the format called `name`."""
    if name not in MX_FORMATS:
        raise ValueError(f"Unknown number format {name!r}; available: {', '.join(MX_FORMATS)}.")
    element, scale = MX_FORMATS[name]
    return DataType(element, scale, name)
