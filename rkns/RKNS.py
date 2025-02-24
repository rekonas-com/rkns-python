from typing import Self
from pathlib import Path

import zarr

from rkns import RKNSAdapter
from rkns.RKNSUtils import ZarrMode, import_string
from rkns.version import __version__


class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is optionally persisted to a Zarr store."""

    def __init__(self, root: zarr.Group, adapter: RKNSAdapter = None) -> None:
        self.root = root
        self.adapter = adapter
        if self.adapter:
            self.adapter.from_src()

    @staticmethod
    def __make_rkns_header() -> dict:
        """Generate header for RKNS file. This should contain information relevant for the
        compatibility between different RKNS versions.
        It must be a JSON-serializable dict."""

        return {"rknsv_version": __version__}

    @classmethod
    def __init_root(self, store) -> zarr.Group:
        """Initialize the Zarr datastructure for the RKNS object."""
        root = (
            zarr.create_group(store=store, overwrite=False) if store else zarr.group()
        )
        root.attrs["rkns_header"] = self.__make_rkns_header()
        root.create_group(name="raw")
        root.create_group(name="rkns")

        return root

    @classmethod
    def open(self, store) -> Self:
        """Open existing file. This will make the raw and rkns groups read-only."""
        # Open root group read-only
        root = zarr.open_group(store=store, mode=ZarrMode.READ_ONLY.value)
        adapter_type_str = root["raw"].attrs["adapter_type"]
        adapter = None
        if adapter_type_str:
            # Dynamically import the adapter class
            adapter = import_string(adapter_type_str)(raw_group=root["raw"])

        return self(root, adapter)

    @classmethod
    def create(self, store=None, adapter_type_str: str = None, **kwargs) -> Self:
        """Create a new RKNS file with an optional adapter."""
        if isinstance(store, str):
            if Path(store).exists():
                raise FileExistsError()
        root = self.__init_root(store)

        # Create adapter
        adapter = None
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

        return self(root, adapter)
