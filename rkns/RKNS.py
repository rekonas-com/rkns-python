import logging
from pathlib import Path

import zarr

from rkns.RKNSUtils import ZarrMode, import_string

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is optionally persisted to a Zarr store."""

    def __init__(self, root, adapter=None):
        self.root = root
        self.adapter = adapter

    def __init_root(store):
        try:
            root = (
                zarr.create_group(store=store, overwrite=False)
                if store
                else zarr.group()
            )
            root.create_group(name="raw")
            root.create_group(name="rkns")
            return root

        except Exception:
            logger.exception("Root group could not be created.")

    @classmethod
    def open(self, store):
        """Open existing file. This will make the raw and rkns groups read-only."""
        try:
            # Open root group read-only
            root = zarr.open_group(store=store, mode=ZarrMode.READ_ONLY.value)
            adapter_type_str = root["raw"].attrs["adapter_type"]
            adapter = None
            if adapter_type_str:
                # Dynamically import the adapter class
                adapter = import_string(adapter_type_str)(raw_group=root["raw"])

            return RKNS(root, adapter)

        except Exception:
            logger.exception("Error opening RKNS file.")

    @classmethod
    def create(self, store=None, adapter_type_str: str = None):
        """Create a new RKNS file with an optional adapter."""
        try:
            if isinstance(store, str):
                if Path(store).exists():
                    raise FileExistsError()
            root = self.__init_root(store)
            adapter = None
            if adapter_type_str:
                adapter = import_string(adapter_type_str)(raw_group=root["raw"])

            return RKNS(root, adapter)

        except Exception:
            logger.exception("Error creating RKNS.")
