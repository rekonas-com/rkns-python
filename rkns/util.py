from __future__ import annotations

import sys
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zarr
import zarr.storage

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from zarr.abc.codec import BaseCodec


def cached_import(module_path: str, class_name: str) -> Any:
    """
    Import a module and return the specified class or attribute.

    From Django: https://docs.djangoproject.com/en/5.1/ref/utils/#module-django.utils.module_loading

    Parameters
    ----------
    module_path : str
        The path of the module to import.
    class_name : str
        The name of the class or attribute to retrieve from the module.

    Returns
    -------
    Any
        The specified class or attribute from the module.
    """
    # Check whether module is loaded and fully initialized.
    if not (
        (module := sys.modules.get(module_path))
        and (spec := getattr(module, "__spec__", None))
        and getattr(spec, "_initializing", False) is False
    ):
        module = import_module(module_path)
    return getattr(module, class_name)


def import_from_string(dotted_path: str) -> Any:
    """
    Import a dotted module path and return the designated class.

    Parameters
    ----------
    dotted_path : str
        The dotted path of the module and class to import.

    Returns
    -------
    Any
        The designated class from the module.

    Raises
    ------
    ImportError
        If the dotted path is not a valid module path or if the module does not
        define the specified attribute or class.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute/class'
            % (module_path, class_name)
        ) from err


class ZarrMode(Enum):
    """Helper enum for Zarr persistence modes."""

    READ_ONLY = "r"
    READ_WRITE = "r+"
    READ_WRITE_CREATE_IF_NOT_EXISTS = "a"
    OVERWRITE = "w"
    CREATE_IF_NOT_EXISTS = "w-"


class RKNSNodeNames(str, Enum):
    """



    Parameters
    ----------
    str
        _description_
    Enum
        _description_
    """

    # Subclassing str to ensure it actually return a string type and not a literal..
    # https://stackoverflow.com/questions/58608361/string-based-enum-in-python
    raw_root = "_raw"
    rkns_root = "rkns"
    frequency_group_prefix = "fg_"
    view = "view"
    history = "history"
    raw_signal = "signal"
    rkns_signal = "signal"
    rkns_signal_minmaxs = "signal_minmaxs"


def check_rkns_validity(rkns_node: zarr.Group | zarr.Array) -> None:
    # TODO: Proper check of file structure of the /rkns node.
    if not isinstance(rkns_node, zarr.Group):
        raise ValueError(f"The child node {RKNSNodeNames.rkns_root=} is not a group. ")


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


def get_target_store(
    path_or_store: zarr.storage.StoreLike | Path | str,
) -> zarr.storage.StoreLike:
    """
    Convert the provided path or store to a valid zarr store.

    Parameters
    ----------
    path_or_store
        Target path or store

    Returns
    -------
    zarr.storage.StoreLike
        The target zarr store
    """
    if isinstance(path_or_store, (str, Path)):
        path = Path(path_or_store)
        if path.exists():
            raise FileExistsError(f"Export target already exists: {path}")
        return zarr.storage.LocalStore(path)
    else:
        # Assume it's already a zarr store
        return path_or_store


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
        # Create the target subgroup
        target_subgroup = target_group.create_group(name)
        # Copy the subgroup attributes
        # Recursively copy the subgroup contents
        copy_group_recursive(subgroup, target_subgroup)
