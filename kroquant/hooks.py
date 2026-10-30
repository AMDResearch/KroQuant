# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Bookkeeping of the forward hooks that the stages put on the model."""

from typing import Any, Protocol


class _Removable(Protocol):
    def remove(self) -> Any: ...


class HookSet:
    """Hook handles (or other `HookSet`s) that are removed together, in insertion order.

    Also a context manager: the hooks are removed when the `with` block exits.
    """

    def __init__(self, *handles: _Removable) -> None:
        self._handles: list[_Removable] = list(handles)

    def add(self, handle: _Removable) -> _Removable:
        self._handles.append(handle)
        return handle

    def remove(self) -> None:
        handles, self._handles = self._handles, []
        for handle in handles:
            handle.remove()

    def __len__(self) -> int:
        return len(self._handles)

    def __enter__(self) -> "HookSet":
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove()
