from enum import Enum
import zarr
import pyedflib
import numpy as np
import math
import json
from pathlib import Path
import functools

from datetime import datetime


class ZarrMode(Enum):
    """Zarr persistence mode."""

    READ_ONLY = "r"
    READ_WRITE = "r+"
    READ_WRITE_CREATE_IF_NOT_EXISTS = "a"
    OVERWRITE = "w"
    CREATE_IF_NOT_EXISTS = "w-"


class RKNS:
    """RKNS (Rekonas) data format.
    It is build on top of a nested structure of Zarr groups and arrays. See __ for full specification.
    /
    ├── annotations
    ├── derived_values
    └── edf_data
        └── edf_data_array
    """

    def __add_creation_metadata(self):
        self._group.attrs["created_on"] = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

    def __init_edf_data_group(self, overwrite):
        self._edf_data_group = self._group.create_group(
            name="edf_data", overwrite=overwrite
        )

    def __init_annotations_group(self, overwrite):
        self._annotations_group = self._group.create_group(
            name="annotations", overwrite=overwrite
        )

    def __init_derived_values_group(self, overwrite):
        self._derived_values_group = self._group.create_group(
            name="derived_values", overwrite=overwrite
        )

    def __init_main_group(self, load: bool, mode: ZarrMode):
        if load and not self.path:
            raise ValueError("Path must be set if load is true.")

        if load:
            self._group = zarr.open_group(store=self.path, mode=mode)

            self._edf_data_group = (
                self._group["edf_data"]
                if self._group.__contains__("edf_data")
                else None
            )
            if self._edf_data_group.__contains__("edf_data_array"):
                self._edf_data_array = self._edf_data_group["edf_data_array"]
                self._edf_data_present = True
                if (
                    self._edf_data_group.attrs.asdict()
                    and self._edf_data_array.attrs.asdict()
                ):
                    self._edf_meta_data_present = True

            self._annotations_group = (
                self._group["annotations"]
                if self._group.__contains__("annotations")
                else None
            )
            self._derived_values_group = (
                self._group["derived_values"]
                if self._group.__contains__("derived_values")
                else None
            )

        else:
            self._group = (
                zarr.group(store=self.path, overwrite=(mode == ZarrMode.OVERWRITE))
                if self.path
                else zarr.group()
            )
            self.__add_creation_metadata()
            self.__init_edf_data_group(overwrite=(mode == ZarrMode.OVERWRITE))
            self.__init_annotations_group(overwrite=(mode == ZarrMode.OVERWRITE))
            self.__init_derived_values_group(overwrite=(mode == ZarrMode.OVERWRITE))

    def __init__(
        self,
        path: str = None,
        load=False,
        mode: ZarrMode = ZarrMode.READ_WRITE_CREATE_IF_NOT_EXISTS,
    ):
        self._edf_meta_data_present = False
        self._edf_data_present = False
        self.path = path
        self.__init_main_group(load, ZarrMode.READ_ONLY if load else mode)

    def tree(self):
        return self._group.tree()

    def attributes(self):
        return self._group.attrs.asdict()

    def edf_header(self):
        return (
            self._edf_data_group.attrs["edf_header"]
            if self._edf_meta_data_present
            else "Does not contain any EDF meta data"
        )

    def edf_channels(self):
        return (
            self._edf_data_array.attrs["edf_channel_headers"]
            if self._edf_meta_data_present
            else "Does not contain any EDF meta data"
        )

    @functools.cached_property
    def rkns_header(self):
        return self._edf_data_array.attrs["rkns_header"]

    @functools.cached_property
    def __get_edf_record_duration(self):
        return int(self.rkns_header["record_duration"])

    @functools.cached_property
    def __get_edf_sample_frequency(self):
        return np.asanyarray(self.rkns_header["sample_frequency"])

    @functools.cached_property
    def __get_edf_chunk_size(self):
        return np.asanyarray(self.rkns_header["chunk_size"])

    @functools.cached_property
    def __get_edf_samples_per_record(self):
        return (
            self.__get_edf_sample_frequency * self.__get_edf_record_duration
        ).astype(np.int16)

    @functools.cached_property
    def __get_edf_record_size(self):
        return self.__get_edf_samples_per_record.sum()

    @functools.cached_property
    def __get_edf_channel_start_idx(self):
        return [
            np.add.reduce(self.__get_edf_samples_per_record[:idx])
            for idx in np.arange(len(self.__get_edf_samples_per_record))
        ]

    @functools.cached_property
    def __get_edf_channel_end_idx(self):
        return self.__get_edf_channel_start_idx + self.__get_edf_samples_per_record

    @functools.cached_property
    def __get_edf_channel_idx(self):
        return np.stack(
            (self.__get_edf_channel_start_idx, self.__get_edf_channel_end_idx), axis=1
        )

    @functools.lru_cache(maxsize=1, typed=False)
    def __get_edf_data_chunk(self, chunk_idx: int):
        """Return an ndarray corresponding to an entire Zarr chunk"""

        chunk_size = self.__get_edf_chunk_size
        chunk_start_idx = chunk_idx * chunk_size
        return self._edf_data_array[chunk_start_idx : chunk_start_idx + chunk_size]

    def get_edf_data_record(self, index: int):
        """Returns a single data record as list of channels.
        No conversion to physical values is made in this step."""

        # To reduce expensive disk I/O for small data records, we always get a full chunk,
        # but use a cahced function.
        record_size = self.__get_edf_record_size
        chunk_size = self.__get_edf_chunk_size

        chunk_idx = (index * record_size) // chunk_size
        record_start_idx = (record_size * index) % chunk_size

        record_data = self.__get_edf_data_chunk(chunk_idx)[
            record_start_idx : record_start_idx + record_size
        ]

        return [record_data[idx[0] : idx[1]] for idx in self.__get_edf_channel_idx]

    def get_edf_data_records(
        self, start: int = 0, stop: int = None, physical: bool = True
    ):
        """Returns EDF data as list of channels. By default, physical values are returned.
        If start and or stop a set to a subset of the data, only that slice is returend.
        """

        rkns_header = self.rkns_header
        number_of_records = int(rkns_header["number_of_records"])

        if start < 0:
            raise ValueError("Start musst be greater than or equal to 0.")
        if start > number_of_records:
            raise ValueError(f"Start musst be smaller than {number_of_records}.")
        if stop and (stop > number_of_records or stop < -number_of_records):
            raise IndexError(
                f"Out of bounds, stop must be smaller than or equal to {number_of_records}."
            )
        if stop and (start > stop and stop > 0):
            raise ValueError("Start must be smaller than or equal to stop.")
        if stop and (stop < 0 and start > (number_of_records + stop)):
            raise ValueError("Start must be smaller than or equal to stop.")

        records = [
            self.get_edf_data_record(record)
            for record in np.arange(
                start=start, stop=stop if stop else number_of_records
            )
        ]

        channels = [
            np.concatenate([record[idx] for record in records])
            for idx in np.arange(len(records[0]))
        ]

        if not physical:
            return channels

        edf_channel_headers = json.loads(self.edf_channels())

        return [
            pyedflib.highlevel.dig2phys(
                channel,
                int(edf_channel_headers[idx]["digital_min"]),
                int(edf_channel_headers[idx]["digital_max"]),
                float(edf_channel_headers[idx]["physical_min"]),
                float(edf_channel_headers[idx]["physical_max"]),
            )
            for idx, channel in enumerate(channels)
        ]

    def edf_data_array(self):
        """Return the EDf data array as stored in the Zarr archive."""
        return (
            self._edf_data_array
            if self._edf_data_present
            else "Does not contain any EDF data"
        )

    @classmethod
    def load(self, path: str):
        """Load RKNS object from file."""

        return RKNS(path, load=True)

    @classmethod
    def from_edf(
        self,
        edf_path: str,
        path: str = None,
        overwrite: bool = False
    ):
        """Returns an RKNS object populated from an EDF file.

        :param edf_path: path to EDF file
        :type edf_path: str
        :param path: path to RKNS
        :type path: str
        :param progress: prints progress to console, defaults to False
        :type progress: bool, optional
        :return: RKNS object with data from given EDF file
        :rtype: rkns.RKNS
        """

        if path and not overwrite and Path(path).exists():
            raise FileExistsError(
                "A file with the specified path already exists. Please set overwrite to true, or set a different path."
            )

        rkns = RKNS(path, mode=ZarrMode.OVERWRITE) if path else RKNS()

        # Use the low-level EdfReader to get essential information about the file
        reader = pyedflib.EdfReader(edf_path)
        record_duration = reader.datarecord_duration
        number_of_records = reader.datarecords_in_file
        # number_of_samples = reader.getNSamples()
        sample_frequency = reader.getSampleFrequencies()
        reader.close()

        samples_per_record = (sample_frequency * record_duration).astype(np.int16)
        record_size = samples_per_record.sum()
        # size_of_zarr_array = np.asanyarray(number_of_samples).sum()

        # We align the chunk size to a multitude of 30s windows.
        # According to the official documentation, zarr performs best with chunk sizes of around 1M.
        # Thus, the size of a chunk is caluclated as follows:
        # chunk size per 30s = 2 (bytes) * 30 (s) * (1 (s) / duration (s)) * number of data points per record
        # chunk size multiplier: the smallest size >= (1 000 000 / size per 30s)
        # optimal chunks size: chunk size multiplier * chunk size per 30s
        chunk_size_per_30 = 2 * 30 * (1.0 / record_duration) * record_size
        chunk_size_multiplier = math.ceil(1000000 / chunk_size_per_30)
        chunk_size = int(chunk_size_multiplier * chunk_size_per_30)

        channel_data, channel_headers, header = pyedflib.highlevel.read_edf(
            edf_path, digital=True
        )

        # TODO: Currently we are writing the entire EDF data to a temporary numpy array which is then passed to zarr.
        # To bound the required memory, we should regularly append partial arrays to the zarr array.
        # Appending small partial arrays to the zarr array degrades I/O performance as zarr flushed data to disk under the hood.
        # array_index = 0
        # tmp_data_array = np.zeros(shape=size_of_zarr_array, dtype=np.int16)
        # for record in np.arange(number_of_records):
        #     for idx, channel in enumerate(channel_data):
        #         num_samples = samples_per_record[idx]
        #         start_idx = record * num_samples
        #         tmp_data_array[array_index : array_index + num_samples] = channel[
        #             start_idx : start_idx + num_samples
        #         ]
        #         array_index += num_samples
        #     if progress:
        #         progress_p = record / number_of_records * 100.0
        #         print(
        #             f"\r\tProgress: {str(format(progress_p, '.2f')).zfill(3)} %", end=""
        #         )
        
        tmp_data_array = np.concatenate(
            [
                np.concatenate(
                    [
                        channel[
                            record_idx * num_samples : record_idx * num_samples
                            + num_samples
                        ]
                        for channel, num_samples in zip(
                            channel_data, samples_per_record
                        )
                    ]
                )
                for record_idx in np.arange(number_of_records)
            ]
        )

        try:
            edf_data_array = rkns._edf_data_group.array(
                data=tmp_data_array,
                name="edf_data_array",
                chunks=chunk_size,
            )
            rkns._edf_data_present = True
            rkns._edf_data_array = edf_data_array

        except IOError as e:
            print("Could not write array.", e)

        # TODO: Preprocess EDF headers to make them JSON serializable.
        # Currently we are wrapping any non-serializable field as a string.
        rkns._edf_data_group.attrs["edf_header"] = json.dumps(header, default=str)
        rkns._edf_data_array.attrs["rkns_header"] = {
            "number_of_records": number_of_records,
            "record_duration": record_duration,
            "sample_frequency": sample_frequency.tolist(),
            "chunk_size": chunk_size,
        }
        edf_data_array.attrs["edf_channel_headers"] = json.dumps(
            channel_headers, default=str
        )
        rkns._edf_meta_data_present = True

        return rkns
