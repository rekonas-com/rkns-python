from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .generics import ZarrArray, ZarrGroup
from .utils_interface import TreeRepr

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class _StoreHandler(ABC):
    """Handles low-level interactions with Zarr storage for RKNS objects."""

    @abstractmethod
    def __init__(self, store: Any | None) -> None:
        pass

    @property
    @abstractmethod
    def root(self) -> ZarrGroup:
        pass

    @property
    @abstractmethod
    def raw(self) -> ZarrGroup:
        pass

    @property
    @abstractmethod
    def rkns(self) -> ZarrGroup:
        pass

    @property
    @abstractmethod
    def signals(self) -> ZarrGroup:
        pass

    @abstractmethod
    def get_channels_by_fg(self, frequency_group: str) -> list[str]:
        pass

    @abstractmethod
    def get_group(self, path: str, mode: str = "r") -> ZarrGroup:
        """Get a Zarr group at the specified path."""
        pass

    @abstractmethod
    def get_array(self, path: str) -> ZarrArray:
        """Get a Zarr array at the specified path."""
        pass

    @abstractmethod
    def create_group(
        self, path: str | None = None, overwrite: bool = False
    ) -> ZarrGroup:
        pass

    @abstractmethod
    def create_hierarchy(
        self,
        root_node: ZarrGroup,
        nodes: Iterable[str],
        *,
        overwrite: bool = False,
    ) -> list[ZarrGroup]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def tree(self, max_depth: int | None = None, show_attrs: bool = True) -> TreeRepr:
        pass

    @abstractmethod
    def export_to_path_or_store(self, path_or_store: Any | Path | str):
        pass

    @abstractmethod
    def deep_compare(
        self,
        other: "_StoreHandler",
        max_depth: int | None = None,
        compare_values: bool = True,
        compare_attributes: bool = True,
    ) -> bool:
        pass
