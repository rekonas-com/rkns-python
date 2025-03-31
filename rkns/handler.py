import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, cast

import zarr
import zarr.storage
from zarr.core.group import GroupMetadata

from rkns.util import RKNSNodeNames, TreeRepr, group_tree_with_attrs_async

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

    from rkns.util import TreeRepr


class StoreHandler:
    """Handles low-level interactions with Zarr storage for RKNS objects."""

    def __init__(self, store: zarr.storage.StoreLike | None) -> None:
        if store is None:
            store = zarr.storage.MemoryStore()
        elif isinstance(store, (Path, str)) and Path(store).suffix == ".zip":
            store = zarr.storage.ZipStore(store, mode="w")
        self._store = store
        self._root: zarr.Group | None = None
        self._raw: zarr.Group | None = None
        self._rkns: zarr.Group | None = None
        self._is_closed = False

    @property
    def root(self) -> zarr.Group:
        if self._root is None:
            try:
                self._root = zarr.open_group(self._store, mode="r+")
            except FileNotFoundError as e:
                raise RuntimeError("The root node ('/') does not exist.") from e
        return self._root

    @property
    def raw(self) -> zarr.Group:
        if self._raw is None:
            try:
                self._raw = zarr.open_group(
                    self._store, path=RKNSNodeNames.raw_root.value, mode="r"
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"The {RKNSNodeNames.rkns_root.value} node does not exist."
                ) from e
        return cast(zarr.Group, self._raw)

    @property
    def rkns(self) -> zarr.Group:
        if self._rkns is None:
            try:
                self._rkns = zarr.open_group(
                    self._store, path=RKNSNodeNames.rkns_root.value, mode="r+"
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"The {RKNSNodeNames.rkns_root.value} node does not exist."
                ) from e
        return self._rkns

    @property
    def signals(self) -> zarr.Group:
        rkns_signal = RKNSNodeNames.rkns_signals_group.value
        return cast(zarr.Group, self.rkns[rkns_signal])

    def get_group(self, path: str, mode: str = "r") -> zarr.Group:
        """Get a Zarr group at the specified path."""
        return zarr.open_group(self._store, path=path, mode=mode)

    def get_array(self, path: str) -> zarr.Array:
        """Get a Zarr array at the specified path."""
        return zarr.open_array(self._store, path=path)

    def create_group(
        self, path: str | None = None, overwrite: bool = False
    ) -> zarr.Group:
        return zarr.create_group(store=self._store, path=path, overwrite=overwrite)

    def create_hierarchy(
        self,
        root_node: zarr.Group,
        nodes: Iterable[str],
        *,
        overwrite: bool = False,
    ):
        _dict: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] = {
            node: GroupMetadata() for node in nodes
        }
        return [node for node in root_node.create_hierarchy(_dict, overwrite=overwrite)]
        #         {
        #     f"{RKNSNodeNames.raw_root.value}": ,
        #     f"{RKNSNodeNames.history.value}": GroupMetadata(),
        #     f"{RKNSNodeNames.popis.value}": GroupMetadata(),
        #     f"{RKNSNodeNames.rkns_root.value}/{RKNSNodeNames.rkns_signals_group.value}": GroupMetadata(),
        #     f"{RKNSNodeNames.rkns_root.value}/{RKNSNodeNames.rkns_annotations_group.value}": GroupMetadata(),
        # }

    # def create_array(self, path: str, data: Any, **kwargs) -> zarr.Array:
    #     """Create a new array at the specified path with given data."""
    #     parent_path = "/".join(path.split("/")[:-1])
    #     array_name = path.split("/")[-1]

    #     parent = self.get_group(parent_path, mode="r+")
    #     return parent.create_array(array_name, data=data, **kwargs)

    # def copy_array(self, source_path: str, dest_path: str, **kwargs) -> zarr.Array:
    #     """Copy an array from source path to destination path."""
    #     data = self.get_array(source_path)[:]
    #     return self.create_array(dest_path, data, **kwargs)

    def close(self) -> None:
        if not self._is_closed:
            try:
                if hasattr(self._store, "close"):
                    self._store.close()  # type: ignore
                self._is_closed = True
            except Exception as e:
                logger.error(f"Error closing store: {str(e)}", exc_info=True)

    def tree(self, max_depth: int | None = None, show_attrs: bool = True) -> TreeRepr:
        return self.root._sync(
            group_tree_with_attrs_async(
                self.root._async_group, max_depth=max_depth, show_attrs=show_attrs
            )
        )
