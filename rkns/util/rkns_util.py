from __future__ import annotations

from enum import Enum

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
    raw_signal = "signal"
    rkns_signal = "signal"
    rkns_signal_minmaxs = "signal_minmaxs"


def check_rkns_validity(rkns_node: zarr.Group | zarr.Array) -> None:
    # TODO: Proper check of file structure of the /rkns node.
    if not isinstance(rkns_node, zarr.Group):
        raise ValueError(f"The child node {RKNSNodeNames.rkns_root=} is not a group. ")
