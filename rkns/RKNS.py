from pathlib import Path

import zarr

from rkns.RKNSUtils import ZarrMode, import_string


class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is optionally persisted to a Zarr store."""

    def __init__(self, root, adapter=None):
        self.root = root
        self.adapter = adapter

    def __init_root(store):
        root = (
            zarr.create_group(store=store, overwrite=False) if store else zarr.group()
        )
        root.create_group(name="raw")
        root.create_group(name="rkns")
        return root

    @classmethod
    def open(self, store):
        """Open existing file. This will make the raw and rkns groups read-only."""
        # Open root group read-only
        root = zarr.open_group(store=store, mode=ZarrMode.READ_ONLY.value)
        adapter_type_str = root["raw"].attrs["adapter_type"]
        adapter = None
        if adapter_type_str:
            # Dynamically import the adapter class
            adapter = import_string(adapter_type_str)(raw_group=root["raw"])

        return RKNS(root, adapter)


    @classmethod
    def create(self, store=None, adapter_type_str: str = None):
        """Create a new RKNS file with an optional adapter."""
        if isinstance(store, str):
            if Path(store).exists():
                raise FileExistsError()
        root = self.__init_root(store)
        adapter = None
        if adapter_type_str:
            adapter = import_string(adapter_type_str)(raw_group=root["raw"])

        return RKNS(root, adapter)
