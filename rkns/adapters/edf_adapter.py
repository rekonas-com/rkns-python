from __future__ import annotations

import tempfile
from collections import defaultdict
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyedflib
import zarr
import zarr.codecs as codecs
import zarr.errors

from rkns.adapters.base import RKNSBaseAdapter
from rkns.file_formats import FileFormat
from rkns.util import RKNSNodeNames, add_child_array, get_freq_group

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# TODO: Move this into a separate (external) config
RAW_CHUNK_SIZE_BYTES = 1024 * 1024 * 8  # 8MB Chunks


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
        signal_headers[i]["frequency_group"] = get_freq_group(
            signal_headers[i]["sample_frequency"]
        )


class RKNSEdfAdapter(RKNSBaseAdapter):
    """RKNS adapter for the EDF format."""

    compressors = codecs.ZstdCodec(level=3)

    def _populate_raw_from_file(
        self, file_path: Path, file_format: FileFormat
    ) -> zarr.Group:
        # TODO this simply loads the whole chunk into memory.
        # this should be doable in a more elegant manner using (variable) chunks
        byte_array = np.fromfile(file_path, dtype=np.byte)

        add_child_array(
            parent_node=self._handler.raw,
            data=byte_array,
            name=RKNSNodeNames.raw_signal.value,
            chunks=RAW_CHUNK_SIZE_BYTES,
            compressors=codecs.ZstdCodec(level=3),
            attributes={
                "filename": file_path.name,
                "format": file_format.value,
                "st_mtime": file_path.stat().st_mtime,
                "md5": md5(byte_array.tobytes()).hexdigest(),
            },
        )

        return self._handler.raw

    def populate_rkns_from_raw(
        self,
        overwrite_if_exists: bool = False,
        validate: bool = True,
    ) -> zarr.Group:
        rkns_node = self._handler.rkns
        rkns_signals_node = self._handler.signals
        raw_signal_node = self._handler.raw[RKNSNodeNames.raw_signal.value]

        # TODO: This is just a hacky workaround to use the existing library.
        # We probably need our custom parser..
        # dump the byte content into a named temporary file and provide the path to pyedflib.
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(raw_signal_node[:].tobytes())  # type: ignore

            filepath = temp_file.name
            channel_data, signal_headers, header = pyedflib.highlevel.read_edf(
                filepath, digital=True
            )  # type: ignore

        add_frequency_groups_to_headers(signal_headers)

        fg_arrays, fg_attributes, rkns_attributes = self._extract_data(
            channel_data, signal_headers, header, validate=validate
        )
        rkns_node.update_attributes(rkns_attributes)

        for fg in fg_arrays.keys():
            fg_node = rkns_signals_node.create_group(fg)
            fg_node.update_attributes(fg_attributes[fg])
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

        return rkns_signals_node

    def _extract_data(
        self, channel_data, signal_headers, header, validate: bool = True
    ):
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
                np.array(p_minmax_d_minmax, dtype=np.float64)
            )
            # build attributes that are per frequency group
            for pyedf_key, rkns_attribute_name in frequency_group_attributes.items():
                fg_attributes[fg][rkns_attribute_name].append(s_header[pyedf_key])
            fg_attributes[fg]["sfreq_Hz"] = s_header["sample_frequency"]

            # build attributes that are per channel, and will be stored as a dict/JSON in /rkns/
            for pyedf_key, rkns_attribute_name in channel_wise_attribute_text.items():
                channel_to_attribute[channel][rkns_attribute_name] = s_header[pyedf_key]
        for fg in fg_arraylist.keys():
            fg_arrays[fg]["signal"] = np.stack(
                fg_arraylist[fg]["signal"], 1, dtype=np.int16
            )
            fg_arrays[fg]["signal_minmaxs"] = np.stack(
                fg_arraylist[fg]["signal_minmaxs"], 1, dtype=np.float64
            )

        header["recording_duration_in_s"] = (
            len(channel_data[0]) / signal_headers[0]["sample_frequency"]
        )
        if validate:
            self.validate_consistent_duration(channel_data, signal_headers)

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
