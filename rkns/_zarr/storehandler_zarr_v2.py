from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

import zarr
import zarr.storage

from rkns.util import RKNSNodeNames

from .generics import ZarrArray, ZarrGroup
from .storehandler_interface import _StoreHandler
from .utils_interface import TreeRepr
from .utils_zarr_v2 import _ZarrV2Utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class StoreHandlerZarrV2(_StoreHandler):
    """
    Handles low-level interactions with Zarr storage for RKNS objects.
    Zarr V2 Version.
    """

    def __init__(self, store: zarr.storage.StoreLike | None) -> None:
        if store is None:
            store = zarr.storage.MemoryStore()
        elif isinstance(store, (Path, str)) and Path(store).suffix == ".zip":
            store = zarr.storage.ZipStore(store, mode="w")
        self._store = store
        self._root: ZarrGroup | None = None
        self._raw: ZarrGroup | None = None
        self._rkns: ZarrGroup | None = None
        self._is_closed = False

    @property
    def root(self) -> ZarrGroup:
        if self._root is None:
            try:
                self._root = zarr.open_group(self._store, mode="a")
            except FileNotFoundError as e:
                raise RuntimeError("The root node ('/') does not exist.") from e
        return self._root

    @property
    def raw(self) -> ZarrGroup:
        if self._raw is None:
            try:
                self._raw = zarr.open_group(
                    self._store, path=RKNSNodeNames.raw_root.value, mode="a"
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"The {RKNSNodeNames.rkns_root.value} node does not exist."
                ) from e
        return cast(ZarrGroup, self._raw)

    @property
    def rkns(self) -> ZarrGroup:
        if self._rkns is None:
            try:
                self._rkns = zarr.open_group(
                    self._store, path=RKNSNodeNames.rkns_root.value, mode="a"
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"The {RKNSNodeNames.rkns_root.value} node does not exist."
                ) from e
        return self._rkns

    @property
    def signals(self) -> ZarrGroup:
        rkns_signal = RKNSNodeNames.rkns_signals_group.value
        return cast(ZarrGroup, self.rkns[rkns_signal])

    def get_channels_by_fg(self, frequency_group: str) -> list[str]:
        return cast(list[str], self.signals[frequency_group].attrs["channels"])

    def get_group(self, path: str, mode: str = "r") -> ZarrGroup:
        """Get a Zarr group at the specified path."""
        return zarr.open_group(self._store, path=path, mode=mode)

    def get_array(self, path: str) -> ZarrArray:
        """Get a Zarr array at the specified path."""
        return zarr.open_array(self._store, path=path)

    def create_group(
        self, path: str | None = None, overwrite: bool = False
    ) -> ZarrGroup:
        mode = "w-" if not overwrite else "w"
        return zarr.open_group(store=self._store, path=path, mode=mode)

    def create_hierarchy(
        self,
        root_node: ZarrGroup,
        nodes: Iterable[str],
        *,
        overwrite: bool = False,
    ) -> list[ZarrGroup]:
        """
        Create a hierarchy of Zarr groups under the given root node.

        Parameters
        ----------
        root_node : ZarrGroup
            The parent group to start the hierarchy from
        nodes : Iterable[str]
            An iterable of paths relative to the root_node, e.g. ["/a/b/c", "/a/d", "/x/y"]
        overwrite : bool, default=False
            If True, will overwrite existing groups, otherwise will use them

        Returns
        -------
        list[ZarrGroup]
            List of all created Zarr groups
        """
        created = []
        for path, name in process_paths(nodes):
            current_group = zarr.open_group(
                store=root_node.store,
                path=root_node.path + path,
                mode="r+",
            )
            new_group = current_group.create_group(name, overwrite=overwrite)
            created.append((root_node.path + path + name, new_group))
            # Normalize path to remove leading/trailing slashes and handle empty paths
            # Start from the root node
            # current_group = root_node

            # # Create each component in the path
            # for component in components:
            #     if component not in created_components:
            #         try:
            #             prev_group = current_group
            #             current_group = current_group.create_group(
            #                 component, overwrite=overwrite
            #             )
            #             created.append(current_group)
            #             created_components.add(prev_group.name + current_group.name)
            #         except Exception as e:
            #             # Handle any errors during group creation
            #             raise ValueError(
            #                 f"Failed to create group '{component}' in path '{path}': {str(e)}"
            #             ) from e
        return created

    def tree(self, max_depth: int | None = None, show_attrs: bool = True) -> TreeRepr:
        return _ZarrV2Utils.group_tree_with_attrs(
            self.root, max_depth=max_depth, show_attrs=show_attrs
        )

    def export_to_path_or_store(self, path_or_store: Any | Path | str):
        target_store = _ZarrV2Utils.get_or_create_target_store(path_or_store)
        if isinstance(target_store, zarr.storage.ZipStore):
            # The current Zarr implementation has issues with ZIP exports...
            # That is, attributes are simply not written.
            raise NotImplementedError()

        try:
            target_root = zarr.group(store=target_store, overwrite=True)
            _ZarrV2Utils.copy_group_recursive(self.root, target_root)
        finally:
            if isinstance(target_store, zarr.storage.ZipStore):
                target_store.close()

    def deep_compare(
        self,
        other: "_StoreHandler",
        max_depth: int | None = None,
        compare_values: bool = True,
        compare_attributes: bool = True,
    ) -> bool:
        return _ZarrV2Utils.deep_compare_groups(
            self.root,
            other.root,
            max_depth=max_depth,
            compare_values=compare_values,
            compare_attributes=compare_attributes,
        )

    def close(self) -> None:
        if not self._is_closed:
            try:
                if hasattr(self._store, "close"):
                    self._store.close()  # type: ignore
                self._is_closed = True
            except Exception as e:
                logger.error(f"Error closing store: {str(e)}", exc_info=True)


def process_paths(paths_list):
    """
    Process a list of paths and return a list of tuples containing the path and basename.
    Includes intermediate path components as separate entries.

    Args:
        paths_list (list): A list of strings representing paths

    Returns:
        list: A list of tuples, where each tuple contains (path, basename)
              If a path doesn't have a directory component, path will be '/'
              All path segments are included as separate entries

    Example:
        >>> process_paths(['_raw', 'history', 'rkns/signals', 'rkns/annotations'])
        [('/', '_raw'), ('/', 'history'), ('/', 'rkns'), ('rkns', 'signals'), ('rkns', 'annotations')]
    """
    result: list[tuple[str, str]] = []
    seen_paths = set()

    for path in paths_list:
        parts = path.split("/")

        # Process each part of the path
        current_path = ""

        for i, part in enumerate(parts):
            if i == 0:
                # First part always has parent path '/'
                parent = "/"
                # Add the tuple if we haven't seen it before
                if (parent, part) not in seen_paths:
                    result.append((parent, part))

                    seen_paths.add((parent, part))
                current_path = part
            else:
                # For subsequent parts, parent is the current_path
                parent = current_path
                # Add the tuple if we haven't seen it before
                if (parent, part) not in seen_paths:
                    result.append((parent, part))
                    seen_paths.add((parent, part))
                # Update current_path for next iteration
                current_path = f"{current_path}/{part}"

    for i, (path, name) in enumerate(result):
        if path != "/":
            result[i] = (f"/{path}/", name)

    return result
