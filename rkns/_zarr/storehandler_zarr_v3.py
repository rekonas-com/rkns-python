from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

import zarr
import zarr.storage
from zarr.core.group import GroupMetadata

from rkns._zarr.utils_zarr_v3 import _ZarrV3Utils
from rkns.util import RKNSNodeNames

from .generics import ZarrArray, ZarrGroup
from .storehandler_interface import _StoreHandler
from .utils_interface import TreeRepr

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata


class StoreHandlerZarrV3(_StoreHandler):
    """Handles low-level interactions with Zarr storage for RKNS objects."""

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
                self._root = zarr.open_group(self._store, mode="r+")
            except FileNotFoundError as e:
                raise RuntimeError("The root node ('/') does not exist.") from e
        return self._root

    @property
    def raw(self) -> ZarrGroup:
        if self._raw is None:
            try:
                self._raw = zarr.open_group(
                    self._store, path=RKNSNodeNames.raw_root.value, mode="r"
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
                    self._store, path=RKNSNodeNames.rkns_root.value, mode="r+"
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
        return zarr.create_group(store=self._store, path=path, overwrite=overwrite)

    def create_hierarchy(
        self,
        root_node: ZarrGroup,
        nodes: Iterable[str],
        *,
        overwrite: bool = False,
    ) -> list[ZarrGroup]:
        _dict: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] = {
            node: GroupMetadata() for node in nodes
        }
        return cast(
            list[ZarrGroup],
            [node for node in root_node.create_hierarchy(_dict, overwrite=overwrite)],
        )
        #         {
        #     f"{RKNSNodeNames.raw_root.value}": ,
        #     f"{RKNSNodeNames.history.value}": GroupMetadata(),
        #     f"{RKNSNodeNames.popis.value}": GroupMetadata(),
        #     f"{RKNSNodeNames.rkns_root.value}/{RKNSNodeNames.rkns_signals_group.value}": GroupMetadata(),
        #     f"{RKNSNodeNames.rkns_root.value}/{RKNSNodeNames.rkns_annotations_group.value}": GroupMetadata(),
        # }

    # def create_array(self, path: str, data: Any, **kwargs) -> ZarrArray:
    #     """Create a new array at the specified path with given data."""
    #     parent_path = "/".join(path.split("/")[:-1])
    #     array_name = path.split("/")[-1]

    #     parent = self.get_group(parent_path, mode="r+")
    #     return parent.create_array(array_name, data=data, **kwargs)

    # def copy_array(self, source_path: str, dest_path: str, **kwargs) -> ZarrArray:
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
        return _ZarrV3Utils.group_tree_with_attrs(
            self.root, max_depth=max_depth, show_attrs=show_attrs
        )

    def export_to_path_or_store(self, path_or_store: Any | Path | str):
        target_store = _ZarrV3Utils.get_or_create_target_store(path_or_store)
        if isinstance(target_store, zarr.storage.ZipStore):
            # The current Zarr implementation has issues with ZIP exports...
            # That is, attributes are simply not written.
            raise NotImplementedError()

        try:
            target_root = zarr.group(store=target_store, overwrite=True)
            _ZarrV3Utils.copy_group_recursive(self.root, target_root)
        finally:
            target_store.close()

    def deep_compare(
        self,
        other: "_StoreHandler",
        max_depth: int | None = None,
        compare_values: bool = True,
        compare_attributes: bool = True,
    ) -> bool:
        return _ZarrV3Utils.deep_compare_groups(
            self.root,
            other.root,
            max_depth=max_depth,
            compare_values=compare_values,
            compare_attributes=compare_attributes,
        )
