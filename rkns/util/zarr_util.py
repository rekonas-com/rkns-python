from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Sequence, cast

import zarr
import zarr.storage
from zarr.abc.store import Store

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy.typing import ArrayLike
    from zarr.abc.codec import BaseCodec

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
        zarr_array.attrs.update(attributes)


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
