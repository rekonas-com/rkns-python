from __future__ import annotations

# rkns/util.py
import io
import os
import sys
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Sequence, cast

import rich
import rich.console
import rich.tree
import zarr
import zarr.storage
from zarr.abc.store import Store
from zarr.core.group import AsyncGroup

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy.typing import ArrayLike
    from zarr.abc.codec import BaseCodec

    T = TypeVar("T", bound=type)


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
    From Django: https://docs.djangoproject.com/en/5.1/ref/utils/#module-django.utils.module_loading

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


def check_open(method: Callable):
    """
    Decorator to check if the RKNS object is closed before executing a method

    Parameters
    ----------
    method
        _description_
    """

    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_is_closed") and self._is_closed:
            raise RuntimeError(
                f"Cannot execute {method.__name__}: RKNS object has been closed"
            )
        return method(self, *args, **kwargs)

    return wrapper


def apply_check_open_to_all_methods(cls: T) -> T:
    """Apply the @check_open decorator to all methods of a class."""
    for name, method in cls.__dict__.items():
        if (
            callable(method)
            and not name.startswith("__")
            and not isinstance(method, (staticmethod, classmethod))
        ):
            # Skip staticmethods and classmethods
            setattr(cls, name, check_open(method))
    return cls


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
