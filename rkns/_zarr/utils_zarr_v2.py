from __future__ import annotations

from pathlib import Path

# Handle ZarrGroup compatibility across versions
# this has to happen before the zarr import.
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, cast

import numpy as np
import zarr.convenience
import zarr.storage
from zarr.attrs import Attributes

from rkns.errors import (
    ArrayShapeMismatchError,
    ArrayValueMismatchError,
    AttributeMismatchError,
    GroupComparisonError,
    MemberCountMismatchError,
    NameMismatchError,
    PathMismatchError,
)

from .generics import ZarrArray, ZarrGroup
from .types import JSON, CodecType, Store

# handle codecs across version
from .utils_interface import ZarrUtils

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy.typing import ArrayLike

    T = TypeVar("T", bound=type)

__all__ = ["_ZarrV2Utils"]


class _ZarrV2Utils(ZarrUtils):
    @staticmethod
    def copy_attributes(
        source: ZarrGroup | ZarrArray, target: ZarrGroup | ZarrArray
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

    @staticmethod
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
        zarr.convenience.copy_all(source=source_group, dest=target_group)

    @staticmethod
    def get_or_create_target_store(
        path_or_store: Store | Path | str, mode: Literal["r", "w", "a"] = "w"
    ) -> Store:
        if isinstance(path_or_store, (str, Path)):
            path = Path(path_or_store)
            if path.exists():
                raise FileExistsError(f"Export target already exists: {path}")
            if path.suffix == ".zip":
                return cast(Store, zarr.storage.ZipStore(path, mode=mode))
            else:
                return cast(
                    Store, zarr.storage.DirectoryStore(path, dimension_separator="/")
                )

        elif isinstance(path_or_store, zarr.storage.Store):
            return cast(Store, path_or_store)
        else:
            raise TypeError(f"Invalid path_or_store={path_or_store}.")

    @staticmethod
    def add_child_array(
        parent_node: ZarrGroup,
        data: ArrayLike,
        name: str,
        attributes: dict[str, Any] | None = None,
        compressors: CodecType | None = None,
        **kwargs,
    ):
        zarr_array = parent_node.create(
            name=name,
            shape=data.shape,  # type: ignore
            dtype=data.dtype,  # type: ignore
            compressor=compressors,  # singular!
            **kwargs,
        )
        zarr_array[:] = data

        if attributes is not None:
            zarr_array.attrs.update(**attributes)

    @staticmethod
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
                if not _ZarrV2Utils.compare_attrs(attr1[key], attr2[key]):
                    return False
            return True
        elif (
            type(attr1) is type(attr2)
            and isinstance(attr1, Iterable)
            and isinstance(attr2, Iterable)
            and not isinstance(attr1, (str, bytes))
        ):
            return len(attr1) == len(attr2) and all(
                a == b for a, b in zip(attr1, attr2)
            )
        else:
            return attr1 == attr2

    @staticmethod
    def deep_compare_groups(
        group1: ZarrGroup,
        group2: ZarrGroup,
        max_depth: int | None = None,
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
        members1 = sorted(
            [x for x in _ZarrV2Utils.iter_zarr_children(group1, max_depth=max_depth)]
        )
        members2 = sorted(
            [x for x in _ZarrV2Utils.iter_zarr_children(group2, max_depth=max_depth)]
        )

        # Check if the number of members is the same
        if len(members1) != len(members2):
            raise MemberCountMismatchError(
                f"Number of members does not match: {len(members1)} vs {len(members2)}"
            )

        if compare_attributes:
            if not _ZarrV2Utils.compare_attrs(group1.attrs, group2.attrs):
                raise AttributeMismatchError(
                    "Attribute values do not match at root node."
                )

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
                if not _ZarrV2Utils.compare_attrs(attrs1, attrs2):
                    raise AttributeMismatchError("Attribute values do not match.")

        return True

    @staticmethod
    def iter_zarr_children(
        group: zarr.Group,
        max_depth: int | None = None,
        current_path: str = "",
        current_depth: int = 0,
    ) -> Iterator[tuple[str, zarr.Group | zarr.Array]]:
        """
        Iterate over all children of a Zarr group as (key, group/array) pairs.

        Args:
            group: The root Zarr group.
            max_depth: Maximum depth to traverse (None for unlimited depth).
            current_path: Internal use for recursion (track hierarchical path).
            current_depth: Internal use for recursion (track current depth).

        Yields:
            (str, Union[Group, Array]): Pairs of (relative key, object).
        """
        for key, obj in group.items():
            full_key = f"{current_path}/{key}" if current_path else key
            yield (full_key, obj)

            # Recurse into subgroups if depth allows
            if isinstance(obj, zarr.Group) and (
                max_depth is None or current_depth < max_depth
            ):
                yield from _ZarrV2Utils.iter_zarr_children(
                    obj,
                    max_depth=max_depth,
                    current_path=full_key,
                    current_depth=current_depth + 1,
                )
