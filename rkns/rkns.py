from __future__ import annotations

import datetime
import logging
import warnings
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Iterable, Optional, OrderedDict, cast

import numpy as np

from rkns._zarr import StoreHandler, ZarrArray
from rkns._zarr.types import JSON
from rkns.adapters.base import RKNSBaseAdapter
from rkns.adapters.registry import AdapterRegistry
from rkns.detectors.registry import FileFormatRegistry
from rkns.errors import RKNSParseError
from rkns.file_formats import FileFormat
from rkns.lazy import LazySignal
from rkns.util import (
    RKNSNodeNames,
    apply_check_open_to_all_methods,
    check_validity,
    get_freq_group,
)
from rkns.version import __version__

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import Self

    from rkns._zarr.utils_interface import TreeRepr


@apply_check_open_to_all_methods
class RKNS:
    """The RKNS class represents a single ExG record of a subject.
    Data is always a Zarr store."""

    def __init__(self, store_handler: StoreHandler, adapter: RKNSBaseAdapter) -> None:
        self.handler = store_handler
        self._is_closed = False

        self.adapter = adapter

    @property
    def patient_info(self) -> JSON:
        return self.handler.rkns.attrs["patient_info"]

    @property
    def admin_info(self) -> JSON:
        return self.handler.rkns.attrs["admin_info"]

    @property
    def channel_info(self) -> JSON:
        return self.handler.rkns.attrs["channel_info"]

    def get_channel_names(self) -> list[str]:
        return [k for k in self.channel_info.keys()]  # type: ignore

    def _get_channel_names_by_fg(self, frequency_group: str) -> list[str]:
        return self.handler.signals[frequency_group].attrs["channels"]  # type: ignore

    def get_frequency_by_channel(self, channel_name: str) -> float:
        fg = self._get_frequencygroup(channel_name)
        return cast(float, self.handler.signals[fg].attrs["sfreq_Hz"])

    def _get_signal_by_freq(self, frequency: float) -> LazySignal:
        return self._get_signal_by_fg(get_freq_group(frequency))

    def get_signal(
        self,
        channels: str | Iterable[str] | None = None,
        sfreq_Hz: float | None = None,
        time_range: tuple[float, float] = (0, np.inf),
    ) -> np.ndarray:
        """Get signal data for specified channels or frequency group.

        Parameters
        ----------
        channels
            Channel name(s) to retrieve. Mutually exclusive with `sfreq_Hz`.
        sfreq_Hz
            Sampling frequency in Hz to retrieve. Mutually exclusive with `channels`.
        time_range : tuple[float, float], optional
            Time range in seconds to retrieve, by default (0, inf), i.e. the whole time frame.
            The time is with respect to the record duration.

        Returns
        -------
            Signal data array for the specified parameters.

        Raises
        ------
        ValueError
            If both channels and frequency are specified, or neither is specified.
            If specified channels belong to different frequency groups.
        """
        if channels is not None and sfreq_Hz is not None:
            raise ValueError("Specify either channels or frequency, not both.")
        elif channels is None and sfreq_Hz is None:
            raise ValueError("Specify one of either channels or frequency.")
        elif channels is None and sfreq_Hz is not None:
            sfreq_Hz = cast(float, sfreq_Hz)
            fg = get_freq_group(sfreq_Hz)

        elif channels is not None and sfreq_Hz is None:
            if isinstance(channels, str):
                channels = [channels]

            fgs = {self._get_frequencygroup(c) for c in channels}
            if len(fgs) != 1:
                raise ValueError("Channels must belong to the same frequency group.")
            fg = next(iter(fgs))
            sfreq_Hz = cast(float, self.handler.signals[fg].attrs["sfreq_Hz"])
        else:
            # Unnecessary but helps pylance.
            raise RuntimeError("Unreachable code reached..")

        row_idx = self.__build_row_idx_from_timerange(
            sfreq_Hz=sfreq_Hz, time_range=time_range
        )
        col_idx = self.__build_col_idx_from_channels(
            channels=channels, frequency_group=fg
        )
        return self._get_signal_by_fg(fg)[row_idx, col_idx]

    def __build_row_idx_from_timerange(
        self,
        sfreq_Hz: float,
        time_range: tuple[float, float] = (0, np.inf),
    ):
        """
        Build a row index slice from a given time range for samples with the given sampling frequency.

        Parameters
        ----------
        sfreq_Hz : float
            Sampling frequency in Hz.
        time_range : tuple[float, float]
            Time range in seconds (start, end). If `None`, defaults to (0, inf).

        Returns
        -------
        slice
            Slice object representing the row indices corresponding to the time range.

        Raises
        ------
        ValueError
            If the end time is not larger than the start time.
        """
        if time_range[0] is None:
            time_range[0] = 0

        if time_range[1] is None:
            time_range[1] = np.inf

        if time_range[1] <= time_range[0]:
            raise ValueError(
                " Invalid time range: {time_range=}. "
                + "End time must be larger than start time."
            )

        start_idx = int(time_range[0] * sfreq_Hz)
        end_time = min(time_range[0] + self.get_recording_duration(), time_range[1])
        end_idx = int(end_time * sfreq_Hz)
        return slice(start_idx, end_idx)

    def __build_col_idx_from_channels(
        self, channels: Iterable[str] | None, frequency_group: str
    ) -> list[int] | EllipsisType:
        """
        Convert channel names to their corresponding column indices for a given frequency group.

        Parameters
        ----------
        channels
            List of channel names to convert to indices. If None, returns `...` (Ellipsis)
            to indicate all channels should be selected.
        frequency_group
            The frequency group to which the channels belong.

        Returns
        -------
            List of column indices corresponding to the given channels, or `...` if no channels are specified.
        """
        if channels is None:
            return ...
        channel_to_index = self.get_channel_order(frequency_group=frequency_group)
        index_order = [channel_to_index[channel] for channel in channels]
        return index_order

    def get_channel_order(
        self, frequency_group: str | None = None, sfreq_in_Hz: float | None = None
    ) -> OrderedDict[str, int]:
        """
        Get the order in which the channels are stored for a given frequency (in Hz) or frequency group.
        Return an ordered dictionary, with the items corresponding to the channel names,
        and the values correspond to their column index in the underlying zarr store.
        This order corresponds to the one you obtain when querying ALL channels of a group with `get_signal`.

        Note that channels are always grouped with other channels of similar frequency.

        The order of the items corresponds to the index.
        E.g. the returned OrderedDict could look like {"F1":0, "F2": 1, "F3":2, ...}

        Parameters
        ----------
        frequency_group, optional
            Specify the wanted frequency_group, by default None
        sfreq_in_Hz, optional
            Specify the wanted frequency in Hz, by default None

        Returns
        -------
            An OrderedDict mapping the channel name to its column index in the stored array.

        """
        if frequency_group is not None and sfreq_in_Hz is not None:
            raise ValueError("Specify either frequency or frequency_group, not both.")
        elif frequency_group is None and sfreq_in_Hz is None:
            raise ValueError("Specify one of either frequency or frequency_group.")
        elif sfreq_in_Hz is not None:
            frequency_group = get_freq_group(freq_in_Hz=sfreq_in_Hz)
        else:
            frequency_group = cast(str, frequency_group)
        channel_names_ordered = self.handler.get_channels_by_fg(frequency_group)
        channel_to_index = OrderedDict(
            (item, idx) for idx, item in enumerate(channel_names_ordered)
        )
        return channel_to_index

    def get_recording_duration(self) -> float:
        """
        Return duration of the recording in seconds.

        Returns
        -------
            Duration in seconds.
        """
        return self.admin_info["recording_duration_in_s"]  # type: ignore

    def get_recording_start(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self.admin_info["recording_date"])  # type: ignore

    def _get_signal_by_fg(self, frequency_group: str) -> LazySignal:
        digital_signal = self._get_digital_signal_by_fg(frequency_group=frequency_group)
        pminmax_dminmax = self._pminmax_dminmax_by_fg(frequency_group=frequency_group)

        l_signal = LazySignal.from_minmaxs(
            digital_signal,
            pmin=cast(np.ndarray, pminmax_dminmax[[0]]),
            pmax=cast(np.ndarray, pminmax_dminmax[[1]]),
            dmin=cast(np.ndarray, pminmax_dminmax[[2]]),
            dmax=cast(np.ndarray, pminmax_dminmax[[3]]),
        )
        return l_signal

    def _get_digital_signal_by_fg(self, frequency_group: str) -> ZarrArray:
        return self.handler.signals[frequency_group][RKNSNodeNames.rkns_signal.value]  # type: ignore

    def _pminmax_dminmax_by_fg(self, frequency_group: str) -> ZarrArray:
        return self.handler.signals[frequency_group][
            RKNSNodeNames.rkns_signal_minmaxs.value
        ]  # type: ignore

    def _get_frequencygroup(self, channel_name: str) -> str:
        # attributes of the /rkns contain the mapping from channel_name to frequency_group
        return self.channel_info[channel_name]["frequency_group"]  # type: ignore

    def _get_frequencygroups(self) -> list[str]:
        return [k for k in self.handler.signals.keys()]

    def is_equal_to(
        self,
        other: RKNS,
        max_depth: Optional[int] = None,
        compare_values: bool = True,
        compare_attributes: bool = True,
    ) -> bool:
        return self.handler.deep_compare(
            other.handler,
            max_depth=max_depth,
            compare_values=compare_values,
            compare_attributes=compare_attributes,
        )

    def export(self, path_or_store: Any | Path | str) -> None:
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

        self.handler.export_to_path_or_store(path_or_store)

    def get_fileformat_of_raw_signal(self) -> FileFormat:
        _raw_signal = self.handler.raw[RKNSNodeNames.raw_signal.value]
        fileformat_id = _raw_signal.attrs["format"]
        return FileFormat(fileformat_id)

    @classmethod
    def from_file(
        cls,
        file_path: Any,
        populate_from_raw: bool = True,
        target_store: Any | None = None,
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
        return RKNSBuilder(target_store).from_file(file_path, populate_from_raw)

    def _reconstruct_original_file(self, file_path: str | Path) -> None:
        signal_array = self.handler.raw[RKNSNodeNames.raw_signal.value]
        # Write the array to the file in binary mode
        with open(file_path, "wb") as file:
            file.write(signal_array[:].tobytes())  # type: ignore

    def populate_rkns_from_raw(
        self, overwrite_if_exists: bool = False, validate: bool = True
    ) -> Self:
        self.adapter.populate_rkns_from_raw(
            overwrite_if_exists=overwrite_if_exists,
            validate=validate,
        )
        return self

    def reset_rkns(self) -> Self:
        return self.populate_rkns_from_raw(overwrite_if_exists=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.handler.close()

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
        return self.handler.tree(max_depth, show_attrs)


class RKNSBuilder:
    def __init__(self, store: Any | None = None):
        self._handler = StoreHandler(store)

    def _init_base_structure(self) -> None:
        """
        Initialize root node with all base attributes (no separate helpers).
        """
        # root node with header + timestamp
        root = self._handler.create_group(path=None)
        root.attrs.update(
            {
                "rkns_header": self._make_rkns_header(),
                "creation_time": datetime.datetime.now().isoformat(),
            }
        )

        # hierarchy of top-level groups
        self._handler.create_hierarchy(
            root_node=root,
            nodes=[
                f"{RKNSNodeNames.raw_root.value}",
                f"{RKNSNodeNames.history.value}",
                f"{RKNSNodeNames.popis.value}",
                f"{RKNSNodeNames.rkns_root.value}/{RKNSNodeNames.rkns_signals_group.value}",
                f"{RKNSNodeNames.rkns_root.value}/{RKNSNodeNames.rkns_annotations_group.value}",
            ],
        )

    @classmethod
    def _make_rkns_header(cls) -> dict[str, str]:
        """Generate header for RKNS file. This should contain information relevant for the
        compatibility between different RKNS versions.
        It must be a JSON-serializable dict."""

        return {"rkns_version": __version__, "rkns_implementation": "python"}

    def from_file(
        self,
        file_path: Any,
        populate_from_raw: bool = True,
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
            rkns = self.from_existing_rkns_store(file_path)
        else:
            if not isinstance(file_path, str):
                # should not be reachable, as Stores will automatically be detected as RKNS.
                raise TypeError(
                    f"For external formats  must be str | Path, but is {file_path=}"
                )
            rkns = self.from_external_format(file_path, file_format=file_format)
            if populate_from_raw:
                rkns.populate_rkns_from_raw()

        return rkns

    def from_existing_rkns_store(
        self,
        store: Any,
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
        handler = StoreHandler(store)

        # compare header versions
        try:
            rkns_header = cast(dict[str, str], handler.root.attrs["rkns_header"])
        except KeyError as e:
            raise RKNSParseError("No rkns_header found in store.") from e

        current_header = self._make_rkns_header()
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
            check_validity(handler.root)

        file_format = FileFormat(handler.raw.attrs["format"])
        Adapter = AdapterRegistry.get_adapter(file_format)
        adapter = Adapter(handler=self._handler)

        return RKNS(store_handler=handler, adapter=adapter)

    def from_external_format(
        self,
        file_path: str | Path,
        file_format: FileFormat,
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
        file_path = Path(file_path)
        self._init_base_structure()

        Adapter = AdapterRegistry.get_adapter(file_format)
        adapter = Adapter(handler=self._handler)
        adapter.populate_raw_from_file(file_path, file_format)

        rkns = RKNS(store_handler=self._handler, adapter=adapter)

        return rkns
