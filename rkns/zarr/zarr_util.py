from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Sequence, cast

import numpy as np
import rich
import rich.console
import rich.tree
import zarr
import zarr.abc.codec
import zarr.registry
import zarr.storage
from zarr.abc.store import Store
from zarr.core.attributes import Attributes
from zarr.core.group import AsyncGroup

from rkns.errors import (
    ArrayShapeMismatchError,
    ArrayValueMismatchError,
    AttributeMismatchError,
    GroupComparisonError,
    MemberCountMismatchError,
    NameMismatchError,
    PathMismatchError,
)

# Handle ZarrGroup compatibility across versions
try:
    # v3
    from zarr import Group as ZarrGroup  # type: ignore
except ImportError:
    # v2
    from zarr.hierarchy import Group as ZarrGroup  # type: ignore  # noqa: F401
from zarr import Array as ZarrArray

# handle codecs across version
from .types import JSON

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeVar

    from numpy.typing import ArrayLike

    try:
        # v3
        from zarr.abc.codec import BaseCodec as CodecType  # type: ignore # noqa: F401
    except ImportError:
        # v2
        from numcodecs.abc import Codec as CodecType  # type: ignore  # noqa: F401
    from zarr import Array as ZarrArray

    T = TypeVar("T", bound=type)


# class _ZarrImplementation(ABC):
#     @staticmethod
#     @abstractmethod
#     def add_child_array(
#         parent_node,  # group
#         data: ArrayLike,
#         name: str,
#         attributes: dict[str, Any] | None = None,
#         compressors: Codec | None = None,
#         **kwargs,
#     ):
#         pass


def add_child_array(
    parent_node: ZarrGroup,
    data: ArrayLike,
    name: str,
    attributes: dict[str, Any] | None = None,
    compressors: zarr.abc.codec.BaseCodec | None = None,
    **kwargs,
):
    zarr_array = parent_node.create_array(
        name=name,
        shape=data.shape,  # type: ignore
        dtype=data.dtype,  # type: ignore
        compressors=compressors,
        **kwargs,
    )
    zarr_array[:] = data

    if attributes is not None:
        zarr_array.update_attributes(attributes)


def get_or_create_target_store(
    path_or_store: Store | Path | str, mode: Literal["r", "w", "a"] = "w"
) -> Store:
    if isinstance(path_or_store, (str, Path)):
        path = Path(path_or_store)
        if path.exists():
            raise FileExistsError(f"Export target already exists: {path}")
        if path.suffix == ".zip":
            return zarr.storage.ZipStore(path, mode=mode)
        else:
            return zarr.storage.LocalStore(path, read_only=mode not in ["w", "a"])
    elif isinstance(path_or_store, Store):
        return path_or_store
    else:
        raise TypeError("Invalid {path_or_store=}.")


def copy_attributes(
    source: ZarrGroup | zarr.Array, target: ZarrGroup | zarr.Array
) -> None:
    """
    Copy all attributes from source to target.

    Parameters
    ----------
    source
        Source group or array
    target
        Target group or array
    """
    for key, value in source.attrs.items():
        target.attrs[key] = value


def copy_group_recursive(source_group: ZarrGroup, target_group: ZarrGroup) -> None:
    """
    Recursively copy a group and all its contents to the target.

    Parameters
    ----------
    source_group
        Source group to copy from
    target_group
        Target group to copy to
    """
    copy_attributes(source_group, target_group)
    for name, array in source_group.arrays():
        target_array = target_group.create_array(
            name=name,
            shape=array.shape,
            dtype=array.dtype,
            chunks=array.chunks,
            compressors=array.compressors,
            fill_value=array.fill_value,
        )

        target_array[:] = array[:]
        copy_attributes(array, target_array)

    # Recursively copy all subgroups
    for name, subgroup in source_group.groups():
        target_subgroup = target_group.create_group(name)
        copy_group_recursive(subgroup, target_subgroup)


def deep_compare_groups(
    group1: ZarrGroup,
    group2: ZarrGroup,
    max_depth: Optional[int] = None,
    compare_values: bool = True,
    compare_attributes: bool = True,
) -> bool:
    """
    Perform a deep comparison of two Group objects.
    Defaults to the complete, most expensive comparison.
    Adjust max_depth, compare_values and compare_attributes for a lighter comparison

    Parameters
    ----------
    group1 : AsyncGroup
        The first async group to compare.
    group2 : AsyncGroup
        The second async group to compare.
    max_depth : Optional[int], optional
        The maximum depth to compare, by default None (unlimited).
    compare_values : bool, optional
        Whether to compare the actual array values, by default False (only compare shapes).
    show_attrs : bool, optional
        Whether to show and compare attributes, by default True.

    Returns
    -------
    bool
        True if the groups are equal.

    Raises
    ------
    GroupComparisonError
        If the groups are not equal, with detailed explanation of the failure.
    """

    if not isinstance(group1, ZarrGroup) or not isinstance(group2, ZarrGroup):
        raise TypeError(
            f"Function not available for types {type(group1)} and {type(group1)}."
        )

    # Preliminary checks
    if group1.name != group2.name:
        raise NameMismatchError(
            f"Group names do not match: '{group1.name}' vs '{group2.name}'"
        )

    # Get members of both groups
    members1 = sorted([x for x in group1.members(max_depth=max_depth)])
    members2 = sorted([x for x in group2.members(max_depth=max_depth)])

    # Check if the number of members is the same
    if len(members1) != len(members2):
        raise MemberCountMismatchError(
            f"Number of members does not match: {len(members1)} vs {len(members2)}"
        )

    if compare_attributes:
        if not compare_attrs(group1.attrs, group2.attrs):
            raise AttributeMismatchError("Attribute values do not match at root node.")

    # Iterate through members simultaneously
    for (key1, node1), (key2, node2) in zip(members1, members2):
        node1_type = type(node1)
        node2_type = type(node2)
        if key1 != key2:
            raise PathMismatchError(f"Keys do not match: '{key1}' vs '{key2}'")
        if node1_type != node2_type:
            raise GroupComparisonError(
                f"Nodes do not match for key '{key1}': {node1} vs {node2}"
            )

        if issubclass(node1_type, ZarrArray):
            node1, node2 = cast(ZarrArray, node1), cast(ZarrArray, node2)
            if node1.shape != node2.shape:
                raise ArrayShapeMismatchError(
                    f"Array shapes do not match for key '{key1}': {node1.shape} vs {node2.shape}"
                )
            if compare_values and not np.allclose(node1[:], node2[:]):
                raise ArrayValueMismatchError(
                    f"Array values do not match for key '{key1}'"
                )
        # elif issubclass(node1_type, AsyncGroup) and (node1.name != node2.name):
        #     raise GroupPathMismatchError(
        #         f"Group paths do not match for key '{key1}': '{node1.name}' vs '{node2.name}'"
        #     )
        # elif node1 != node2:
        #     breakpoint()
        #     raise GroupComparisonError(
        #         f"Nodes do not match for key '{key1}': {node1} vs {node2}"
        #     )

        if compare_attributes:
            attrs1 = node1.attrs
            attrs2 = node2.attrs
            if attrs1.keys() != attrs2.keys():
                raise AttributeMismatchError(
                    f"Attribute keys do not match for key '{key1}': {attrs1.keys()} vs {attrs2.keys()}"
                )
            if not compare_attrs(attrs1, attrs2):
                raise AttributeMismatchError("Attribute values do not match.")

    return True


def compare_attrs(attr1: JSON, attr2: JSON) -> bool:
    """
    Helper function to compare attributes, including nested dictionaries.

    Parameters
    ----------
    attr1 : Any
        The first attribute to compare.
    attr2 : Any
        The second attribute to compare.

    Returns
    -------
    bool
        True if the attributes are equal, False otherwise.
    """
    if isinstance(attr1, Attributes):
        attr1 = dict(attr1)
    if isinstance(attr2, Attributes):
        attr2 = dict(attr2)
    if isinstance(attr1, dict) and isinstance(attr2, dict):
        if attr1.keys() != attr2.keys():
            return False
        for key in attr1.keys():
            if not compare_attrs(attr1[key], attr2[key]):
                return False
        return True
    elif (
        type(attr1) is type(attr2)
        and isinstance(attr1, Iterable)
        and isinstance(attr2, Iterable)
        and not isinstance(attr1, (str, bytes))
    ):
        return len(attr1) == len(attr2) and all(a == b for a, b in zip(attr1, attr2))
    else:
        return attr1 == attr2


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


async def _group_tree_with_attrs_async(
    group: AsyncGroup, max_depth: int | None = None, show_attrs: bool = True
) -> TreeRepr:
    """
    Return a tree representation of the group.
    Added some information in addition to the zarr-python function this was based on.


    Adapted from https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/_tree.py

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

    Parameters
    ----------
    group
        _description_
    max_depth, optional
        _description_, by default None
    show_attrs, optional
        _description_, by default True

    Returns
    -------
        _description_
    """
    tree = rich.tree.Tree(label=f"[bold]{group.name}[/bold]")
    nodes = {"": tree}
    members = sorted([x async for x in group.members(max_depth=max_depth)])

    for key, node in members:
        if key.count("/") == 0:
            parent_key = ""
        else:
            parent_key = key.rsplit("/", 1)[0]
        parent = nodes[parent_key]

        # We want what the spec calls the node "name", the part excluding all leading
        # /'s and path segments. But node.name includes all that, so we build it here.
        name = key.rsplit("/")[-1]
        if isinstance(node, AsyncGroup):
            label = f"[bold]{name}[/bold]"
        else:
            label = f"[bold]{name}[/bold] {node.shape} {node.dtype}"

        node_tree = parent.add(label)

        # Our addition to the zarr-python reference
        if show_attrs and len(node.attrs) > 0:
            attr_tree = node_tree.add("[italic][Attributes][/italic]")
            for attr_key in node.attrs.keys():
                attr_value = node.attrs[attr_key]
                value_descr = type(attr_value)
                if issubclass(value_descr, Iterable):
                    value_descr = f"{value_descr} (length: {len(attr_value)})"  # type: ignore
                attr_label = f"[italic]{attr_key}[/italic]: {value_descr}"
                attr_tree.add(attr_label)

        nodes[key] = node_tree

    return TreeRepr(tree)


async def group_tree_with_attrs_async(
    group: AsyncGroup, max_depth: int | None = None, show_attrs: bool = True
) -> TreeRepr:
    return await _group_tree_with_attrs_async(
        group=group, max_depth=max_depth, show_attrs=show_attrs
    )


def get_codec(id: str, *args, **kwargs):
    return zarr.registry.get_codec_class(id)(*args, **kwargs)
