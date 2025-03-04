from __future__ import annotations

from hashlib import md5
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, cast

import numpy as np
import zarr
import zarr.storage

from rkns.adapters.registry import AdapterRegistry
from rkns.detectors.registry import FileFormatRegistry
from rkns.file_formats import FileFormat
from rkns.util import RKNSNodeNames, ZarrMode, import_from_string
from rkns.version import __version__

if TYPE_CHECKING:
    from typing import Any, Self

    from zarr.storage import StoreLike

    from rkns.adapters.base import RKNSBaseAdapter


class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is optionally persisted to a Zarr store."""

    def __init__(self, store: zarr.storage.StoreLike, adapter: RKNSBaseAdapter) -> None:
        self.store = store
        self._root = None
        self.adapter = adapter

    def _get_root(self) -> zarr.Group:
        if self._root is None:
            self._root = cast(zarr.Group, zarr.open(self.store, mode="r+"))
        return self._root

    def _get_raw(self) -> zarr.Group:
        _raw_root = self._get_root()[RKNSNodeNames.raw_root.value]
        return cast(zarr.Group, _raw_root)

    def _get_raw_signal(self) -> zarr.Array:
        _raw_signal = self._get_raw()[RKNSNodeNames.raw_signal.value]
        return cast(zarr.Array, _raw_signal)

    @classmethod
    def from_file(
        cls,
        file_path: StoreLike,
        populate_from_raw: bool = True,
        target_store: StoreLike | None = None,
    ) -> Self:
        """
        Create an instance of RKNS based on a given file path.

        Parameters
        ----------
        file_path
            Filepath or Zarr Store to load file from.
        populate_from_raw
            Whether to directly populate the RKNS file structure from the
            raw data, by default True. Ignored, if the input file_path is already a
            rkns file.

        Returns
        -------
            Instance of the RKNS class.

        Raises
        ------
        NotImplementedError
            If the file format was not recognized.
        """
        file_format = FileFormatRegistry.detect_fileformat(file_path)

        if file_format == FileFormat.UNKNOWN:
            raise NotImplementedError(
                f"File format could not be detected for: {file_path=}. "
                + "This may be due to the file lacking proper suffix, being corrupted, "
                + " or the file format simply not being supported.."
            )

        if file_format == FileFormat.RKNS:
            rkns = cls._from_existing_rkns_store(file_path)
        else:
            if not isinstance(file_path, str):
                # should not be reachable, as Stores will automatically be detected as RKNS.
                raise TypeError(
                    f"For external formats  must be str | Path, but is {file_path=}"
                )
            rkns = cls._from_external_format(
                file_path, file_format=file_format, target_store=target_store
            )
            if populate_from_raw:
                rkns.populate_rkns_from_raw()

        return rkns

    @classmethod
    def _from_existing_rkns_store(
        cls,
        store: zarr.storage.StoreLike,
    ) -> Self:
        # TODO.
        # Should also infer the adapter by itself, since it should be
        # inferrable by the raw node
        raise NotImplementedError()

    @classmethod
    def _from_external_format(
        cls,
        file_path: str | Path,
        file_format: FileFormat,
        target_store: StoreLike | None,
    ) -> Self:
        """
        Create /_raw group, fills it with binary data given in file_path,
        and then stores the file as an array within /_raw/signal.
        The metadata of /_raw/signal are:
        - filename (e.g. "myfile.edf")
        - inferred fileformat (Integer based on FileFormat enum)
        - Date of source file

        Parameters
        ----------
        file_path
            _description_
        file_format
            _description_
            _description_
        target_store
            By default None. If None, a new Memory Store will be instantiated.

        Returns
        -------
            _description_
        """
        if target_store is None:
            target_store = zarr.storage.MemoryStore()
        adapter = AdapterRegistry.get_adapter(file_format=file_format)

        file_path = Path(file_path)
        root_node = cls.__init_root(target_store)

        # fill raw node
        _raw_node = root_node.create_group(name=RKNSNodeNames.raw_root.value)
        # TODO this simply loads the whole chunk into memory.
        # this should be doable in a more elegant manner using (variable) chunks
        byte_array = np.fromfile(file_path, dtype=np.uint8)
        _raw_signal = _raw_node.create_array(
            name=RKNSNodeNames.raw_signal.value,
            shape=byte_array.shape,
            dtype=byte_array.dtype,
        )
        _raw_signal[:] = byte_array
        _raw_signal.attrs["filename"] = file_path.name
        _raw_signal.attrs["format"] = file_format.value
        stat = file_path.stat()
        _raw_signal.attrs["modification_time"] = stat.st_mtime
        _raw_signal.attrs["md5"] = md5(byte_array.tobytes()).hexdigest()

        rkns = cls(store=target_store, adapter=adapter)
        return rkns

    def _reconstruct_original_file(self, file_path: str | Path) -> None:
        signal_array = self._get_raw_signal()
        # Write the array to the file in binary mode
        with open(file_path, "wb") as file:
            file.write(signal_array[:].tobytes())  # type: ignore

    def populate_rkns_from_raw(self, overwrite_existing: bool = False) -> Self:
        raise NotImplementedError()

    def reset_rkns(self) -> Self:
        return self.populate_rkns_from_raw(overwrite_existing=True)

    @classmethod
    def __init_root(cls, store: StoreLike | None) -> zarr.Group:
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
        root.attrs["creation_time"] = time()  # current_time_since_epoch

        return root

    @staticmethod
    def __make_rkns_header() -> dict[str, str]:
        """Generate header for RKNS file. This should contain information relevant for the
        compatibility between different RKNS versions.
        It must be a JSON-serializable dict."""

        return {"rkns_version": __version__, "rkns_implementation": "python"}

    # @classmethod
    # def open(cls, store: StoreLike) -> Self:
    #     """Open existing file. This will make the raw and rkns groups read-only."""
    #     # Open root group read-only
    #     root = zarr.open_group(store=store, mode=ZarrMode.READ_ONLY.value)
    #     adapter_type_str = str(root[RKNSNodeNames.raw_root].attrs["adapter_type"])
    #     adapter = None
    #     if adapter_type_str:
    #         # Dynamically import the adapter class
    #         adapter = import_from_string(adapter_type_str)(
    #             raw_group=root[RKNSNodeNames.raw_root]
    #         )

    #     return cls(root, adapter)

    # @classmethod
    # def create(
    #     cls,
    #     adapter_type_str: str,
    #     store: StoreLike = zarr.storage.MemoryStore(),
    #     **kwargs: dict[str, Any],
    # ) -> Self:
    #     """Create a new RKNS file with an optional adapter."""
    #     root = cls.__init_root(store)
    #     # Create adapter
    #     if adapter_type_str:
    #         # Extract adapter-specific arguments
    #         adapter_args = {
    #             kw.removeprefix("adapter_"): kwargs[kw]
    #             for kw in kwargs
    #             if kw.startswith("adapter_")
    #         }
    #         adapter = import_from_string(adapter_type_str)(
    #             raw_group=root["raw"], **adapter_args
    #         )
    #     else:
    #         adapter = None

    #     return cls(root, adapter)
