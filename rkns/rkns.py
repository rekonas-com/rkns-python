from __future__ import annotations

import logging
import warnings
from hashlib import md5
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import zarr
import zarr.codecs as codecs
import zarr.errors
import zarr.storage

from rkns.adapters.registry import AdapterRegistry
from rkns.detectors.registry import FileFormatRegistry
from rkns.file_formats import FileFormat
from rkns.util import (
    RKNSNodeNames,
    RKNSParseError,
    apply_check_open_to_all_methods,
    check_rkns_validity,
    copy_attributes,
    copy_group_recursive,
    get_or_create_target_store,
)
from rkns.version import __version__

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.store import Store
    from zarr.storage import StoreLike, StorePath


RAW_CHUNK_SIZE_BYTES = 1024 * 1024 * 8  # 8MB Chunks


@apply_check_open_to_all_methods
class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is always a Zarr store."""

    def __init__(self, store: zarr.storage.StoreLike) -> None:
        self.store = store
        self._root = None
        self._raw = None
        self._is_closed = False

        self.adapter = AdapterRegistry.get_adapter(
            file_format=self.get_fileformat_raw_signal()
        )

    @staticmethod
    def _make_rkns_header() -> dict[str, str]:
        """Generate header for RKNS file. This should contain information relevant for the
        compatibility between different RKNS versions.
        It must be a JSON-serializable dict."""

        return {"rkns_version": __version__, "rkns_implementation": "python"}

    def export(self, path_or_store: Store | Path | str) -> None:
        """
        Export the RKNS object to a new store, creating a deep copy of all data.
        NOTE: This is a temporary solution, as in the future zarr.copy_all should be available.

        Parameters
        ----------
        path_or_store
            Target location or store for the exported RKNS data.
            Can be a path string, Path object, or zarr store.

        Returns
        -------
        RKNS
            A new RKNS instance pointing to the exported data.

        Notes
        -----
        This creates a complete copy of the entire RKNS structure, including
        all arrays, groups and their metadata.
        """
        target_store = get_or_create_target_store(path_or_store)

        if isinstance(target_store, zarr.storage.ZipStore):
            # The current Zarr implementation has issues with ZIP exports...
            # That is, attributes are simply not written.
            raise NotImplementedError()

        try:
            target_root = zarr.group(store=target_store, overwrite=True)
            copy_group_recursive(self._get_root(), target_root)
        finally:
            target_store.close()
        # return RKNS(store=target_store)

    def _get_root(self) -> zarr.Group:
        if self._root is None:
            self._root = zarr.open_group(self.store, mode="r+")
        return self._root

    def _get_raw(self) -> zarr.Group:
        if self._raw is None:
            # NOTE: Read only does not seem to work for individual groups
            # This would have to be done for the whole store.
            self._raw = zarr.open_group(
                self.store, path=RKNSNodeNames.raw_root.value, mode="r"
            )
        return self._raw

    def _get_rkns(self) -> zarr.Group:
        if self._rkns is None:
            self._rkns = zarr.open_group(
                self.store, path=RKNSNodeNames.raw_root.value, mode="r+"
            )
        return self._rkns

    def _get_raw_signal(self) -> zarr.Array:
        _raw_signal = self._get_raw()[RKNSNodeNames.raw_signal.value]
        return cast(zarr.Array, _raw_signal)

    def get_fileformat_raw_signal(self) -> FileFormat:
        fileformat_id = self._get_raw_signal().attrs["format"]
        return FileFormat(fileformat_id)

    @classmethod
    def from_file(
        cls,
        file_path: StoreLike,
        populate_from_raw: bool = True,
        target_store: StoreLike | None = None,
    ) -> "RKNS":
        """
        Convenience method to create an RKNS instance from a file.
        Delegates to RKNSBuilder.from_file.

        Parameters
        ----------
        file_path
            _description_
        populate_from_raw, optional
            _description_, by default True
        target_store, optional
            _description_, by default None

        Returns
        -------
            _description_
        """
        return RKNSBuilder.from_file(file_path, populate_from_raw, target_store)

    def _reconstruct_original_file(self, file_path: str | Path) -> None:
        signal_array = self._get_raw_signal()
        # Write the array to the file in binary mode
        with open(file_path, "wb") as file:
            file.write(signal_array[:].tobytes())  # type: ignore

    def populate_rkns_from_raw(
        self, overwrite_if_exists: bool = False, validate: bool = True
    ) -> Self:
        self._rkns = self.adapter.populate_rkns_from_raw(
            raw_node=self._get_raw(),
            root_node=self._get_root(),
            overwrite_if_exists=overwrite_if_exists,
            validate=validate,
        )
        return self

    def reset_rkns(self) -> Self:
        return self.populate_rkns_from_raw(overwrite_if_exists=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if hasattr(self.store, "close"):
            self.close()  # type: ignore

    def close(self) -> None:
        """Safely close the underlying store if required"""
        if not self._is_closed:
            try:
                if hasattr(self.store, "close"):
                    self._store.close()  # type: ignore
                self._is_closed = True
            except Exception as e:
                logger.error(f"Error closing store: {str(e)}", exc_info=True)


class RKNSBuilder:
    @classmethod
    def from_file(
        cls,
        file_path: StoreLike,
        populate_from_raw: bool = True,
        target_store: StoreLike | None = None,
    ) -> RKNS:
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
            rkns = cls.from_existing_rkns_store(file_path)
        else:
            if not isinstance(file_path, str):
                # should not be reachable, as Stores will automatically be detected as RKNS.
                raise TypeError(
                    f"For external formats  must be str | Path, but is {file_path=}"
                )
            rkns = cls.from_external_format(
                file_path, file_format=file_format, target_store=target_store
            )
            if populate_from_raw:
                rkns.populate_rkns_from_raw()

        return rkns

    @classmethod
    def from_existing_rkns_store(
        cls,
        store: zarr.storage.StoreLike,
        validate: bool = True,
    ) -> RKNS:
        """
        Create an instance of RKNS from an existing Zarr store.

        Parameters
        ----------
        store : zarr.storage.StoreLike
            Zarr store containing the RKNS data.
        validate : bool, optional
            Whether to validate the RKNS structure, by default True.

        Returns
        -------
        RKNS
            Instance of the RKNS class.

        Raises
        ------
        ValueError
            If the RKNS version in the header does not match the current version.
        """
        if (isinstance(store, Path) or isinstance(store, str)) and Path(
            store
        ).suffix == ".zip":
            root = zarr.group(zarr.storage.ZipStore(store, mode="w"))
        else:
            root = zarr.open(store, mode="r")
        # Check the RKNS header
        try:
            rkns_header = cast(dict[str, str], root.attrs["rkns_header"])
        except KeyError as e:
            raise RKNSParseError("No rkns_header found in store.") from e

        current_header = RKNS._make_rkns_header()
        if rkns_header.get("rkns_version") != current_header["rkns_version"]:
            raise ValueError(
                f"RKNS version mismatch. Expected {current_header['version']}, "
                f"but found {rkns_header.get('version')}."
            )

        # Courtesy warning if another implementation was used for creating the file
        if (
            rkns_header.get("rkns_implementation")
            != current_header["rkns_implementation"]
        ):
            warnings.warn(
                f"The RKNS file was generated by a different implementation ({rkns_header.get('rkns_implementation')}). "
                "Full feature overlap is not guaranteed.",
                UserWarning,
            )

        if validate:
            check_rkns_validity(root)

        return RKNS(store=store)

    @classmethod
    def from_external_format(
        cls,
        file_path: str | Path,
        file_format: FileFormat,
        target_store: StoreLike | None,
    ) -> RKNS:
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

        file_path = Path(file_path)
        root_node = cls.__create_root_node(target_store)

        raw_node = root_node.create_group(name=RKNSNodeNames.raw_root.value)
        cls.__fill_raw_binary(raw_node, file_path, file_format)

        rkns = RKNS(store=target_store)

        return rkns

    @classmethod
    def __fill_raw_binary(
        cls, _raw_node: zarr.Group, file_path: Path, file_format: FileFormat
    ):
        # TODO this simply loads the whole chunk into memory.
        # this should be doable in a more elegant manner using (variable) chunks
        byte_array = np.fromfile(file_path, dtype=np.uint8)

        # TODO: Decide on codec here.
        compressors = codecs.ZstdCodec(level=3)
        _raw_signal = _raw_node.create_array(
            name=RKNSNodeNames.raw_signal.value,
            shape=byte_array.shape,
            dtype=byte_array.dtype,
            chunks=RAW_CHUNK_SIZE_BYTES,
            compressors=compressors,
        )
        _raw_signal[:] = byte_array
        _raw_signal.attrs["filename"] = file_path.name
        _raw_signal.attrs["format"] = file_format.value
        stat = file_path.stat()
        _raw_signal.attrs["modification_time"] = stat.st_mtime
        _raw_signal.attrs["md5"] = md5(byte_array.tobytes()).hexdigest()

    @classmethod
    def __create_root_node(cls, store: StoreLike | None) -> zarr.Group:
        """Initialize the Zarr datastructure for the RKNS object."""

        if isinstance(store, str):
            if Path(store).exists():
                raise FileExistsError(
                    f"Failed to create RKNS file. Path {store} already exists."
                )

        root = (
            zarr.create_group(store=store, overwrite=False) if store else zarr.group()
        )
        root.attrs["rkns_header"] = RKNS._make_rkns_header()
        root.attrs["creation_time"] = time()  # current_time_since_epoch

        return root
