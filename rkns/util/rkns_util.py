from __future__ import annotations

from enum import Enum
from typing import Any, cast

import numpy as np

from rkns._zarr import ZarrGroup


class RKNSNodeNames(str, Enum):
    # Subclassing str to ensure it actually return a string type and not a literal..
    # https://stackoverflow.com/questions/58608361/string-based-enum-in-python
    raw_root = "_raw"
    rkns_root = "rkns"
    rkns_signals_group = "signals"
    rkns_annotations_group = "annotations"
    frequency_group_prefix = "fg_"
    view = "view"
    history = "history"
    popis = "popis"
    raw_signal = "signal"
    rkns_signal = "signal"
    rkns_signal_minmaxs = "signal_minmaxs"


def check_validity(root_node: ZarrGroup | Any) -> None:
    expected_root_attributes = ["rkns_header", "creation_time"]
    for attribute in expected_root_attributes:
        if attribute not in root_node.attrs:
            raise ValueError(f"Missing required root attribute: {attribute=}")

    expected_header_attributes = ["rkns_version", "rkns_implementation"]

    for attribute in expected_header_attributes:
        if attribute not in root_node.attrs["rkns_header"]:  # type: ignore
            raise ValueError(f"Missing required header attribute: {attribute=}")

    expected_root_groups = [
        f"/{RKNSNodeNames.raw_root.value}",
        f"/{RKNSNodeNames.rkns_root.value}",
        f"/{RKNSNodeNames.history.value}",
        f"/{RKNSNodeNames.popis.value}",
    ]

    for expected_group in expected_root_groups:
        if expected_group not in root_node:
            raise ValueError(f"Missing {expected_group=} in root node. Invalid Format.")

    check_rkns_validity(rkns_node=root_node[RKNSNodeNames.rkns_root.value])
    check_raw_validity(_raw_node=root_node[RKNSNodeNames.raw_root.value])


def check_rkns_validity(rkns_node: ZarrGroup | Any) -> None:
    """
    TODO:
    - Check attributes
    - Check array dtypes
    - check array shapes

    """
    if not isinstance(rkns_node, ZarrGroup):
        raise TypeError(f"The root node must be a Group, but is {type(rkns_node)}.")

    if not rkns_node.basename == RKNSNodeNames.rkns_root.value:
        raise ValueError(
            f"The RKNS node basename must be '{RKNSNodeNames.rkns_root.value}', but found '{rkns_node.basename}'"
        )

    # Check that the signals and annotations groups exist.
    expected_children = [
        f"/{RKNSNodeNames.rkns_signals_group.value}",
        f"/{RKNSNodeNames.rkns_annotations_group.value}",
    ]
    for expected_group in expected_children:
        if expected_group not in rkns_node:
            raise ValueError(f"Missing {expected_group=}. Invalid Format.")
        elif not isinstance(rkns_node[expected_group], ZarrGroup):
            raise TypeError(
                f"Expected {expected_group} to be a ZarrGroup, but it is {type(rkns_node[expected_group])}."
            )

    # check that all child groups of the /rkns/signals group start with "fg_"
    # If they do, check that they fg_ groups contain the "signal" and "signal_minmax" arrays.
    rkns_signals_group = cast(
        ZarrGroup, rkns_node[RKNSNodeNames.rkns_signals_group.value]
    )
    for name, subgroup in rkns_signals_group.groups():
        if not subgroup.basename.startswith(RKNSNodeNames.frequency_group_prefix.value):
            raise ValueError(
                f"Group '{subgroup.name}' does not have the prefix {RKNSNodeNames.frequency_group_prefix.value}"
            )
        else:
            if RKNSNodeNames.rkns_signal.value not in subgroup:
                raise ValueError(
                    f"Frequency group '{subgroup.basename}' is missing required '{RKNSNodeNames.rkns_signal.value}' array"
                )

            if RKNSNodeNames.rkns_signal_minmaxs.value not in subgroup:
                raise ValueError(
                    f"Frequency group '{subgroup.basename}' is missing required '{RKNSNodeNames.rkns_signal_minmaxs.value}' array"
                )


def check_raw_validity(_raw_node: ZarrGroup | Any) -> None:
    """
    TODO:
    - Check attribute types
    """
    if not isinstance(_raw_node, ZarrGroup):
        raise TypeError(f"The root node must be a Group, but is {type(_raw_node)}.")

    if not _raw_node.basename == RKNSNodeNames.raw_root.value:
        raise ValueError(
            f"The raw node basename must be '{RKNSNodeNames.raw_root.value}', but found '{_raw_node.basename}'."
        )

    expected_children = [f"/{RKNSNodeNames.raw_signal.value}"]

    for expected_group in expected_children:
        if expected_group not in _raw_node:
            raise ValueError(f"Missing {expected_group=}. Invalid Format.")

    _raw_signal = _raw_node[RKNSNodeNames.raw_signal.value]
    expected_raw_signal_attributes = ["filename", "format", "st_mtime", "md5"]
    for attribute in expected_raw_signal_attributes:
        if attribute not in _raw_signal.attrs:
            raise ValueError(f"Missing required raw signal attribute: {attribute}")

    if not isinstance(_raw_signal, ZarrGroup) and _raw_signal.dtype != np.byte:  # type: ignore
        raise TypeError(
            f"Expected Array of dtype np.byte, but got {type(_raw_signal)}."
        )


def get_freq_group(freq_in_Hz: float) -> str:
    prefix = RKNSNodeNames.frequency_group_prefix.value
    return f"{prefix}{np.round(freq_in_Hz, 1)}"
