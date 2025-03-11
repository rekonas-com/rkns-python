from __future__ import annotations

from enum import Enum
from logging import root
from typing import Any

import numpy as np
import zarr
import zarr.storage


class RKNSNodeNames(str, Enum):
    # Subclassing str to ensure it actually return a string type and not a literal..
    # https://stackoverflow.com/questions/58608361/string-based-enum-in-python
    raw_root = "_raw"
    rkns_root = "rkns"
    frequency_group_prefix = "fg_"
    view = "view"
    history = "history"
    popis = "popis"
    raw_signal = "signal"
    rkns_signal = "signal"
    rkns_signal_minmaxs = "signal_minmaxs"


def check_validity(root_node: zarr.Group | Any) -> None:
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


def check_rkns_validity(rkns_node: zarr.Group | Any) -> None:
    """
    TODO:
    - Check attributes
    - Check array dtypes
    - check array shapes

    """
    if not isinstance(rkns_node, zarr.Group):
        raise TypeError(f"The root node must be a Group, but is {type(rkns_node)}.")

    if not rkns_node.basename == RKNSNodeNames.rkns_root.value:
        raise ValueError(
            f"The RKNS node basename must be '{RKNSNodeNames.rkns_root.value}', but found '{rkns_node.basename}'"
        )

    for name, subgroup in rkns_node.groups():
        # groups starting with fg_ must be frequency groups containing signal and minmax scale
        if subgroup.basename.startswith(RKNSNodeNames.frequency_group_prefix.value):
            if RKNSNodeNames.rkns_signal.value not in subgroup:
                raise ValueError(
                    f"Frequency group '{subgroup.basename}' is missing required '{RKNSNodeNames.rkns_signal.value}' array"
                )

            if RKNSNodeNames.rkns_signal_minmaxs.value not in subgroup:
                raise ValueError(
                    f"Frequency group '{subgroup.basename}' is missing required '{RKNSNodeNames.rkns_signal_minmaxs.value}' array"
                )


def check_raw_validity(_raw_node: zarr.Group | Any) -> None:
    """
    TODO:
    - Check attribute types
    """
    if not isinstance(_raw_node, zarr.Group):
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
    expected_raw_signal_attributes = ["filename", "format", "modification_time", "md5"]
    for attribute in expected_raw_signal_attributes:
        if attribute not in _raw_signal.attrs:
            raise ValueError(f"Missing required raw signal attribute: {attribute}")

    if not isinstance(_raw_signal, zarr.Array) and _raw_signal.dtype != np.byte:  # type: ignore
        raise TypeError(
            f"Expected Array of dtype np.byte, but got {type(_raw_signal)}."
        )
