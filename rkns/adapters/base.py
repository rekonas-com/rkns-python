from __future__ import annotations

from abc import ABC, abstractmethod

import zarr


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
