from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    cast,
)

import numpy as np
import zarr
import zarr.storage
from zarr import Array, AsyncArray, AsyncGroup, Group
from zarr.abc.store import Store
from zarr.core.attributes import Attributes

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeVar

    from numpy.typing import ArrayLike
    from zarr.abc.codec import BaseCodec
    from zarr.core.common import JSON

    T = TypeVar("T", bound=type)


class ZarrMode(Enum):
    """Helper enum for Zarr persistence modes."""

    READ_ONLY = "r"
    READ_WRITE = "r+"
    READ_WRITE_CREATE_IF_NOT_EXISTS = "a"
    OVERWRITE = "w"
    CREATE_IF_NOT_EXISTS = "w-"


def add_child_array(
    parent_node: zarr.Group,
    data: ArrayLike,
    name: str,
    attributes: dict[str, Any] | None = None,
    compressors: BaseCodec | None = None,
):
    zarr_array = parent_node.create_array(
        name=name,
        shape=data.shape,  # type: ignore
        dtype=data.dtype,  # type: ignore
        compressors=compressors,
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
    source: zarr.Group | zarr.Array, target: zarr.Group | zarr.Array
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


def copy_group_recursive(source_group: zarr.Group, target_group: zarr.Group) -> None:
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


class RKNSParseError(Exception):
    pass


class GroupComparisonError(Exception):
    """Base exception for group comparison failures."""

    pass


class NameMismatchError(GroupComparisonError):
    """Raised when group names do not match."""

    pass


class MemberCountMismatchError(GroupComparisonError):
    """Raised when the number of members in groups do not match."""

    pass


class PathMismatchError(GroupComparisonError):
    """Raised when keys (paths) of members do not match."""

    pass


class ArrayShapeMismatchError(GroupComparisonError):
    """Raised when array shapes do not match."""

    pass


class ArrayValueMismatchError(GroupComparisonError):
    """Raised when array values do not match."""

    pass


class GroupPathMismatchError(GroupComparisonError):
    """Raised when nested group names do not match."""

    pass


class AttributeMismatchError(GroupComparisonError):
    """Raised when attributes do not match."""

    pass


def deep_compare_groups(
    group1: Group,
    group2: Group,
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

    if not isinstance(group1, Group) or not isinstance(group2, Group):
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

        if issubclass(node1_type, Array):
            node1, node2 = cast(Array, node1), cast(Array, node2)
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
