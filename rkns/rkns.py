from __future__ import annotations

import logging
import warnings
from hashlib import md5
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, cast

import numpy as np
import zarr
import zarr.codecs as codecs
import zarr.core
import zarr.core.common
import zarr.errors
import zarr.storage

from rkns.adapters.registry import AdapterRegistry
from rkns.detectors.registry import FileFormatRegistry
from rkns.file_formats import FileFormat
from rkns.util import (
    RKNSNodeNames,
    RKNSParseError,
    apply_check_open_to_all_methods,
    check_validity,
    copy_attributes,
    copy_group_recursive,
    get_freq_group,
    get_or_create_target_store,
    group_tree_with_attrs_async,
)
from rkns.util.zarr_util import deep_compare_groups
from rkns.version import __version__

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.store import Store
    from zarr.storage import StoreLike, StorePath

    from rkns.util import TreeRepr

RAW_CHUNK_SIZE_BYTES = 1024 * 1024 * 8  # 8MB Chunks


@apply_check_open_to_all_methods
class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is always a Zarr store."""

    def __init__(self, store: zarr.storage.StoreLike) -> None:
        self.store = store
        self.__root: zarr.Group | None = None
        self.__raw: zarr.Group | None = None
        self.__rkns: zarr.Group | None = None
        self._is_closed = False

        self.adapter = AdapterRegistry.get_adapter(
            file_format=self.get_fileformat_raw_signal()
        )

    @property
    def patient_info(self) -> zarr.core.common.JSON:
        return self._rkns.attrs["patient_info"]

    @property
    def admin_info(self) -> zarr.core.common.JSON:
        return self._rkns.attrs["admin_info"]

    @property
    def channel_info(self) -> zarr.core.common.JSON:
        return self._rkns.attrs["channel_info"]

    def get_channel_names(self) -> list[str]:
        return [k for k in self.channel_info.keys()]  # type: ignore

    def _get_channel_names_by_fg(self, frequency_group: str) -> list[str]:
        return self._signals[frequency_group].attrs["channels"]  # type: ignore

    def get_frequency_by_channel(self, channel_name: str) -> float:
        fg = self._get_frequencygroup(channel_name)
        return self._signals[fg].attrs["sample_frequency_HZ"]  # type: ignore

    def get_signal_by_freq(self, frequency: float) -> np.ndarray:
        return self._get_signal_by_fg(get_freq_group(frequency))

    def _get_signal_by_fg(self, frequency_group: str) -> np.ndarray:
        digital_signal = self._get_digital_signal_by_fg(frequency_group=frequency_group)
        pminmax_dminmax = self._pminmax_dminmax_by_fg(frequency_group=frequency_group)

        # NOTE: Important to cast here: Number might and probably is twice as large as np.int16
        pmin = pminmax_dminmax[[0]]
        pmax = pminmax_dminmax[[1]]
        dmin = pminmax_dminmax[[2]]
        dmax = pminmax_dminmax[[3]]
        m = (pmax - pmin) / (dmax - dmin)
        bias = pmax / m - dmax
        return m * (digital_signal + bias)

    def _get_digital_signal_by_fg(self, frequency_group: str) -> np.ndarray:
        return self._signals[frequency_group][RKNSNodeNames.rkns_signal.value]  # type: ignore

    def _pminmax_dminmax_by_fg(self, frequency_group: str) -> np.ndarray:
        return self._signals[frequency_group][RKNSNodeNames.rkns_signal_minmaxs.value]  # type: ignore

    def _get_frequencygroup(self, channel_name: str) -> str:
        # attributes of the /rkns contain the mapping from channel_name to frequency_group
        return self.channel_info[channel_name]["frequency_group"]  # type: ignore

    def _get_frequencygroups(self) -> list[str]:
        return [k for k in self._signals.keys()]

    def is_equal_to(
        self,
        other: RKNS,
        max_depth: Optional[int] = None,
        compare_values: bool = True,
        compare_attributes: bool = True,
    ) -> bool:
        return deep_compare_groups(
            self._root,
            other._root,
            max_depth=max_depth,
            compare_values=compare_values,
            compare_attributes=compare_attributes,
        )

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
            copy_group_recursive(self._root, target_root)
        finally:
            target_store.close()

    def get_fileformat_raw_signal(self) -> FileFormat:
        _raw_signal = self._raw[RKNSNodeNames.raw_signal.value]
        fileformat_id = _raw_signal.attrs["format"]
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
        signal_array = self._raw[RKNSNodeNames.raw_signal.value]
        # Write the array to the file in binary mode
        with open(file_path, "wb") as file:
            file.write(signal_array[:].tobytes())  # type: ignore

    def populate_rkns_from_raw(
        self, overwrite_if_exists: bool = False, validate: bool = True
    ) -> Self:
        self.adapter.populate_rkns_from_raw(
            raw_node=self._raw,
            root_node=self._root,
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

    @property
    def tree(self, max_depth: int | None = None, show_attrs: bool = True) -> TreeRepr:
        """
        Provide a tree-like overview of the underlying zarr structure.
        Includes groups, array shapes and types, and top-level keys of attributes.

        Parameters
        ----------
        max_depth, optional
            Max depth to be shown, by default None
        show_attrs, optional
            Whether attributes should be shown, by default True

        Returns
        -------
            TODO: Not exactly sure what it
        """
        return self._root._sync(
            group_tree_with_attrs_async(
                self._root._async_group, max_depth=max_depth, show_attrs=show_attrs
            )
        )

    @staticmethod
    def _make_rkns_header() -> dict[str, str]:
        """Generate header for RKNS file. This should contain information relevant for the
        compatibility between different RKNS versions.
        It must be a JSON-serializable dict."""

        return {"rkns_version": __version__, "rkns_implementation": "python"}

    @property
    def _root(self) -> zarr.Group:
        if self.__root is None:
            self.__root = zarr.open_group(self.store, mode="r+")
        return self.__root

    @property
    def _raw(self) -> zarr.Group:
        if self.__raw is None:
            # NOTE: Read only does not seem to work for individual groups
            # This would have to be done for the whole store.
            self.__raw = zarr.open_group(
                self.store, path=RKNSNodeNames.raw_root.value, mode="r"
            )
        return self.__raw

    @property
    def _rkns(self) -> zarr.Group:
        if self.__rkns is None:
            self.__rkns = zarr.open_group(
                self.store, path=RKNSNodeNames.rkns_root.value, mode="r+"
            )
        return self.__rkns

    @property
    def _signals(self) -> zarr.Group:
        rkns_signal = RKNSNodeNames.rkns_signals_group.value
        return cast(zarr.Group, self._rkns[rkns_signal])


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
            root = zarr.open(store, mode="r+")
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
            check_validity(root)

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

        # create history and popis groups
        # TODO: We did not yet define the structure..
        root_node.create_group(name=RKNSNodeNames.history.value)
        root_node.create_group(name=RKNSNodeNames.popis.value)

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
