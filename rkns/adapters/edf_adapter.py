from __future__ import annotations

import tempfile
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pyedflib
import zarr
import zarr.codecs as codecs
import zarr.errors

from rkns.adapters.base import RKNSBaseAdapter
from rkns.util import RKNSNodeNames, add_child_array

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# dictionaries mapping the signal header keys to the keys within RKNS
## These will be added as an additional array of shape (n_channels, len(dictionary)) in /rkns/scaling_array,
## as they are seldom requested individually.
## This will allow a very simple rescaling using numpy broadcasting.
minmax_array_columnorder = [
    "physical_min",
    "physical_max",
    "digital_min",
    "digital_max",
]

## These will be added as a dictionary within the /rkns attributes,
## as this is often of interest for individual channels.
## The key will be the channel name
channel_wise_attribute_text = {
    "dimension": "physical_dimension",
    "transducer": "transducer",
    "prefilter": "prefiltering",
    # this is a custom attribute that I need to compute, and is not given by pyedflib.
    # I list it here to have a nicer implementation                                                                                                                         below.
    "frequency_group": "frequency_group",
}


frequency_group_attributes = {
    "label": "channels",
}


header_patientinfo_attributes = {
    "patientname": "name",
    "patient_additional": "additional",
    "patientcode": "patientcode",
    "sex": "sex",
    "birthdate": "birthdate",
}
header_admininfo_attributes = {
    "admincode": "admincode",
    "technician": "technician",
    "startdate": "recording_date",
    # this is a custom attribute that I need to compute, and is not given by pyedflib.
    # I list it here to have a nicer implementation
    "recording_duration_in_s": "recording_duration_in_s",
}
###########


def add_frequency_groups_to_headers(signal_headers: list[dict[str, Any]]) -> None:
    # loop through the pyedf signal headers and pre-compute the frequency groups
    # based on the sample frequency
    for i in range(len(signal_headers)):
        rounded_frequency = np.round(signal_headers[i]["sample_frequency"], 1)
        signal_headers[i]["frequency_group"] = f"fg_{rounded_frequency}"


class RKNSEdfAdapter(RKNSBaseAdapter):
    """RKNS adapter for the EDF format."""

    compressors = codecs.ZstdCodec(level=3)

    @classmethod
    def populate_rkns_from_raw(
        cls,
        raw_node: zarr.Group,
        root_node: zarr.Group,
        overwrite_if_exists: bool = False,
        validate: bool = True,
    ) -> zarr.Group:
        rkns_node = cls.create_rkns_group(root_node, overwrite_if_exists)
        rsignal_node = raw_node[RKNSNodeNames.raw_signal.value]

        # TODO: This is just a hacky workaround to use the existing library.
        # We probably need our custom parser..
        # dump the byte content into a named temporary file and provide the path to pyedflib.
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(rsignal_node[:].tobytes())  # type: ignore

            filepath = temp_file.name
            channel_data, signal_headers, header = pyedflib.highlevel.read_edf(
                filepath, digital=True
            )  # type: ignore

        add_frequency_groups_to_headers(signal_headers)

        fg_arrays, fg_attributes, rkns_attributes = cls.extract_data(
            channel_data, signal_headers, header, validate=validate
        )
        rkns_node.attrs.update(rkns_attributes)

        for fg in fg_arrays.keys():
            fg_node = rkns_node.create_group(fg)
            fg_node.attrs.update(fg_attributes[fg])
            add_child_array(
                parent_node=fg_node,
                data=fg_arrays[fg]["signal"],
                name="signal",
                attributes={"rows": "samples", "columns": "channels"},
            )
            add_child_array(
                parent_node=fg_node,
                data=fg_arrays[fg]["signal_minmaxs"],
                name="signal_minmaxs",
                attributes={"rows": "channels", "columns": minmax_array_columnorder},
            )

        return rkns_node

    @classmethod
    def extract_data(cls, channel_data, signal_headers, header, validate: bool = True):
        """
        Helper function to extract the data in a format easily translatable to RKNS.
        """

        # infer groups based on sample frequency.
        # These will identify child arrays of /rkns and contain the actual data.
        # the key specifies the name of the array.
        fg_arraylist: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        fg_arrays: dict[str, dict[str, ArrayLike]] = defaultdict(dict)

        # will be stored in  /rkns/fg_1.0, /rkns/fg_500.0, ... attributes
        fg_attributes: dict[str, Any] = defaultdict(lambda: defaultdict(list))

        # will be stored in /rkns attributes
        channel_to_attribute: dict[str, Any] = defaultdict(dict)

        # iterate over the channels
        # a.) group data by frequency
        # b.) remember which channel label maps to which frequency group
        # c.) collect metadata that should belong into frequency groups.
        for idx, s_header in enumerate(signal_headers):
            fg = s_header["frequency_group"]
            channel = s_header["label"]

            # build signal list for Array /rkns/signal
            fg_arraylist[fg]["signal"].append(channel_data[idx])

            # build scaling list for Array /rkns/signal_minmaxs
            p_minmax_d_minmax = [
                s_header[pyedf_key] for pyedf_key in minmax_array_columnorder
            ]
            fg_arraylist[fg]["signal_minmaxs"].append(
                np.array(p_minmax_d_minmax, dtype=np.int16)
            )

            # build attributes that are per frequency group
            for pyedf_key, rkns_attribute_name in frequency_group_attributes.items():
                fg_attributes[fg][rkns_attribute_name].append(s_header[pyedf_key])
            fg_attributes[fg]["sample_frequency_HZ"] = s_header["sample_frequency"]

            # build attributes that are per channel, and will be stored as a dict/JSON in /rkns/
            for pyedf_key, rkns_attribute_name in channel_wise_attribute_text.items():
                channel_to_attribute[channel][rkns_attribute_name] = s_header[pyedf_key]

        for fg in fg_arraylist.keys():
            for key in fg_arraylist[fg].keys():
                fg_arrays[fg][key] = np.stack(fg_arraylist[fg][key], 1, dtype=np.int16)

        header["recording_duration_in_s"] = (
            len(channel_data[0]) / signal_headers[0]["sample_frequency"]
        )
        if validate:
            cls.validate_consistent_duration(channel_data, signal_headers)

        rkns_attributes = dict(patient_info={}, admin_info={})
        for pyedf_key, rkns_attribute_name in header_patientinfo_attributes.items():
            attr = header[pyedf_key]
            if isinstance(attr, datetime):
                attr = attr.isoformat()
            rkns_attributes["patient_info"][rkns_attribute_name] = attr

        for pyedf_key, rkns_attribute_name in header_admininfo_attributes.items():
            attr = header[pyedf_key]
            if isinstance(attr, datetime):
                attr = attr.isoformat()
            rkns_attributes["admin_info"][rkns_attribute_name] = attr

        rkns_attributes["channel_info"] = dict(channel_to_attribute)
        return fg_arrays, fg_attributes, rkns_attributes

    @classmethod
    def validate_consistent_duration(cls, channel_data, signal_headers):
        durations = []
        for channel_idx in range(len(signal_headers)):
            n_samples = len(channel_data[channel_idx])
            durations.append(
                n_samples / signal_headers[channel_idx]["sample_frequency"]
            )
        if not np.all(np.isclose(durations[0], durations)):
            raise ValueError(
                "Channels in the input file are "
                + " inconsistent with respect to the duration of the record."
            )

    # def from_src(self) -> None:
    #     if self.src_path:
    #         """
    #         -------
    #         signals : np.ndarray or list
    #             the signals of the chosen channels contained in the EDF.
    #         signal_headers : list
    #             one signal header for each channel in the EDF.
    #         header : dict
    #             the main header of the EDF file containing meta information.

    #         """

    #         channel_data, signal_headers, header = pyedflib.highlevel.read_edf(
    #             self.src_path, digital=True
    #         )
    #         # TODO: Override highlevel EDFReader to also output filetype
    #         # For now use the underlying EDFReader and extract filtype manually
    #         file_type = -1
    #         with pyedflib.EdfReader(self.src_path) as f:
    #             file_type = f.filetype

    #         # We can hard-code the adapter_type string as it is defined by the concrete implementation
    #         self.raw_group.attrs["adapter_type"] = "rkns.RKNSAdapter.RKNSEdfAdapter"
    #         self.raw_group.attrs["file_type"] = file_type
    #         self.raw_group.attrs["header"] = json.dumps(header, default=str)

    #         signal_data_group = self.raw_group.create_group(name="signal_data")
    #         signal_data_group.attrs["signal_headers"] = json.dumps(
    #             signal_headers, default=str
    #         )
    #         for channel, header in zip(channel_data, signal_headers):
    #             z = signal_data_group.create_array(
    #                 name=header["label"], shape=channel.shape, dtype=channel.dtype
    #             )
    #             z[:] = channel
    #     else:
    #         raise TypeError("src_path cannot be None")

    # def recreate_src(self, path: str) -> None:
    #     file_type = int(cast(int, self.raw_group.attrs["file_type"]))
    #     header = json.loads(cast(str, self.raw_group.attrs["header"]))
    #     # re-create header dict from JSON-serialized header
    #     if header["startdate"]:
    #         header["startdate"] = datetime.datetime.strptime(
    #             header["startdate"], "%Y-%m-%d %H:%M:%S"
    #         )
    #     signal_data_group = self.raw_group["signal_data"]
    #     signal_headers = json.loads(
    #         cast(str, signal_data_group.attrs["signal_headers"])
    #     )
    #     signal_data = [
    #         signal_data_group[signal_header["label"]][:]  # type: ignore
    #         for signal_header in signal_headers
    #     ]
    #     pyedflib.highlevel.write_edf(
    #         edf_file=path,
    #         signals=signal_data,
    #         signal_headers=signal_headers,
    #         header=header,
    #         digital=True,
    #         file_type=file_type,
    #     )
