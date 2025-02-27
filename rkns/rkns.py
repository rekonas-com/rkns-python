from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import zarr
import zarr.storage

from rkns.rkns_util import ZarrMode, import_string
from rkns.version import __version__

if TYPE_CHECKING:
    from typing import Any, Self

    from zarr.storage import StoreLike

    from rkns.rkns_adapter import RKNSBaseAdapter


class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is optionally persisted to a Zarr store."""

    def __init__(
        self, root: zarr.Group, adapter: RKNSBaseAdapter | None = None
    ) -> None:
        self.root = root
        self.adapter = adapter
        if self.adapter:
            self.adapter.from_src()

    @staticmethod
    def __make_rkns_header() -> dict[str, Any]:
        """Generate header for RKNS file. This should contain information relevant for the
        compatibility between different RKNS versions.
        It must be a JSON-serializable dict."""

        return {"rknsv_version": __version__}

    @classmethod
    def __init_root(cls, store: StoreLike) -> zarr.Group:
        """Initialize the Zarr datastructure for the RKNS object."""

        if isinstance(store, str):
            if Path(store).exists():
                raise FileExistsError(
                    f"Failed to create RKNS file. Path {store} already exists."
                )

        root = (
            zarr.create_group(store=store, overwrite=False) if store else zarr.group()
        )
        root.attrs["rkns_header"] = cls.__make_rkns_header()
        root.create_group(name="raw")
        root.create_group(name="rkns")

        return root

    @classmethod
    def open(cls, store: StoreLike) -> Self:
        """Open existing file. This will make the raw and rkns groups read-only."""
        # Open root group read-only
        root = zarr.open_group(store=store, mode=ZarrMode.READ_ONLY.value)
        adapter_type_str = str(root["raw"].attrs["adapter_type"])
        adapter = None
        if adapter_type_str:
            # Dynamically import the adapter class
            adapter = import_string(adapter_type_str)(raw_group=root["raw"])

        return cls(root, adapter)

    @classmethod
    def create(
        cls,
        adapter_type_str: str,
        store: StoreLike = zarr.storage.MemoryStore(),
        **kwargs: dict[str, Any],
    ) -> Self:
        """Create a new RKNS file with an optional adapter."""
        root = cls.__init_root(store)
        # Create adapter
        if adapter_type_str:
            # Extract adapter-specific arguments
            adapter_args = {
                kw.removeprefix("adapter_"): kwargs[kw]
                for kw in kwargs
                if kw.startswith("adapter_")
            }
            adapter = import_string(adapter_type_str)(
                raw_group=root["raw"], **adapter_args
            )
        else:
            adapter = None

        return cls(root, adapter)
