import datetime
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import pyedflib
import zarr

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike


class RKNSBaseAdapter(ABC):
    """Abstract base class for RKNS adapters.
    An adapter can import compatible ExG data from other data formats into an internal raw format.
    The original source file can be recreated from the RKNS raw format.
    Adapters also need to implement a one-way conversion to the RKNS format (no export from RKNS to arbitrary formats).
    """

    def __init__(self, raw_group: zarr.Group, src_path: str | None = None) -> None:
        self.raw_group = raw_group
        self.src_path = src_path

    @abstractmethod
    def from_src(self) -> None:
        """Load data from source format into a Zarr group containing the raw original data."""
        pass

    @abstractmethod
    def recreate_src(self, path: str) -> None:
        """Export the raw original data to its original format. This should exactely rematerialize the original file."""
        pass

    @abstractmethod
    def to_rkns(self) -> None:
        """Transform raw original data into the RKNS format."""
        pass


class RKNSEdfAdapter(RKNSBaseAdapter):
    """RKNS adapter for the EDF format."""

    def __init__(self, raw_group: zarr.Group, src_path: str | None = None) -> None:
        super().__init__(raw_group=raw_group, src_path=src_path)

    def from_src(self) -> None:
        if self.src_path:
            """
            -------
            signals : np.ndarray or list
                the signals of the chosen channels contained in the EDF.
            signal_headers : list
                one signal header for each channel in the EDF.
            header : dict
                the main header of the EDF file containing meta information.

            """

            channel_data, signal_headers, header = pyedflib.highlevel.read_edf(
                self.src_path, digital=True
            )
            # TODO: Override highlevel EDFReader to also output filetype
            # For now use the underlying EDFReader and extract filtype manually
            file_type = -1
            with pyedflib.EdfReader(self.src_path) as f:
                file_type = f.filetype

            # We can hard-code the adapter_type string as it is defined by the concrete implementation
            self.raw_group.attrs["adapter_type"] = "rkns.RKNSAdapter.RKNSEdfAdapter"
            self.raw_group.attrs["file_type"] = file_type
            self.raw_group.attrs["header"] = json.dumps(header, default=str)

            signal_data_group = self.raw_group.create_group(name="signal_data")
            signal_data_group.attrs["signal_headers"] = json.dumps(
                signal_headers, default=str
            )
            for channel, header in zip(channel_data, signal_headers):
                z = signal_data_group.create_array(
                    name=header["label"], shape=channel.shape, dtype=channel.dtype
                )
                z[:] = channel
        else:
            raise TypeError("src_path cannot be None")

    def recreate_src(self, path: str) -> None:
        file_type = int(cast(int, self.raw_group.attrs["file_type"]))
        header = json.loads(cast(str, self.raw_group.attrs["header"]))
        # re-create header dict from JSON-serialized header
        if header["startdate"]:
            header["startdate"] = datetime.datetime.strptime(
                header["startdate"], "%Y-%m-%d %H:%M:%S"
            )
        signal_data_group = self.raw_group["signal_data"]
        signal_headers = json.loads(
            cast(str, signal_data_group.attrs["signal_headers"])
        )
        signal_data = [
            signal_data_group[signal_header["label"]][:]  # type: ignore
            for signal_header in signal_headers
        ]
        pyedflib.highlevel.write_edf(
            edf_file=path,
            signals=signal_data,
            signal_headers=signal_headers,
            header=header,
            digital=True,
            file_type=file_type,
        )

    def to_rkns() -> None:
        # TODO: Implement once RKNS is specified.
        pass
