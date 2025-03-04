from __future__ import annotations

from typing import TYPE_CHECKING

from ..util import import_from_string

if TYPE_CHECKING:
    from ..file_formats import FileFormat
    from .base import RKNSBaseAdapter


class AdapterRegistry:
    """
    Registry for modular addition and lazy loading of new adapters.
    Adapters are registered in the modules __init__.

    Note: This is implemented via a class to simulate a singleton.
    """

    # dictionary that keeps track of module paths
    _adapters: dict[FileFormat, str] = dict()

    @classmethod
    def register(cls, file_format: FileFormat, adapter_path: str) -> None:
        """Register adapter as a string path to defer import."""
        cls._adapters[file_format] = adapter_path

    @classmethod
    def get_adapter(cls, file_format: FileFormat) -> RKNSBaseAdapter:
        """Dynamically load the adapter class only when needed."""
        adapter_path = cls._adapters.get(file_format)
        if not adapter_path:
            raise ValueError(f"No adapter found for file type: {file_format}")

        return import_from_string(adapter_path)
