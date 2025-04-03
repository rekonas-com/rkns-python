"""
This file serves as a generic interface to zarr utilities independent of its
major version (v2 / v3).

This is done via an adapter-like approach, with an abstract interface serving as the
visible API from within the rest of RKNS.
"""

from __future__ import annotations

import io
import os

# Handle ZarrGroup compatibility across versions
# this has to happen before the zarr import.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, cast

import rich
import rich.tree

from .generics import ZarrArray, ZarrGroup

# handle codecs across version
from .types import JSON

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy.typing import ArrayLike

    from .types import CodecType, Store

    T = TypeVar("T", bound=type)


class ZarrUtils(ABC):
    @staticmethod
    @abstractmethod
    def add_child_array(
        parent_node,  # group
        data: ArrayLike,
        name: str,
        attributes: dict[str, Any] | None = None,
        compressors: CodecType | None = None,
        **kwargs,
    ):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def update_attributes(node: ZarrGroup | ZarrArray, attribute_dict: dict):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_or_create_target_store(
        path_or_store: Any | Path | str, mode: Literal["r", "w", "a"] = "w"
    ) -> Store:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def copy_attributes(
        source: ZarrGroup | ZarrArray, target: ZarrGroup | ZarrArray
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def copy_group_recursive(source_group: ZarrGroup, target_group: ZarrGroup) -> None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def deep_compare_groups(
        group1: ZarrGroup,
        group2: ZarrGroup,
        max_depth: Optional[int] = None,
        compare_values: bool = True,
        compare_attributes: bool = True,
    ) -> bool:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def compare_attrs(attr1: JSON, attr2: JSON) -> bool:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def group_tree_with_attrs(
        group: ZarrGroup, max_depth: int | None = None, show_attrs: bool = True
    ) -> TreeRepr:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_codec(id: str, *args, **kwargs) -> CodecType:
        raise NotImplementedError()


class TreeRepr:
    """
    A simple object with a tree-like repr for the Zarr Group.

    Note that this object and it's implementation isn't considered part
    of Zarr's public API.

    Taken from https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/_tree.py

    The MIT License (MIT)

    Copyright (c) 2015-2024 Zarr Developers <https://github.com/zarr-developers>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    """

    def __init__(self, tree: rich.tree.Tree) -> None:
        self._tree = tree

    def __repr__(self) -> str:
        color_system = os.environ.get(
            "OVERRIDE_COLOR_SYSTEM", rich.get_console().color_system
        )
        console = rich.console.Console(file=io.StringIO(), color_system=color_system)  # type: ignore
        console.print(self._tree)
        _console_file = cast(io.StringIO, console.file)
        return str(_console_file.getvalue())

    def _repr_mimebundle_(
        self,
        include: Sequence[str],
        exclude: Sequence[str],
        **kwargs: Any,
    ) -> dict[str, str]:
        # For jupyter support.
        # Unsure why mypy infers the return type to by Any
        return self._tree._repr_mimebundle_(include=include, exclude=exclude, **kwargs)  # type: ignore[no-any-return]
