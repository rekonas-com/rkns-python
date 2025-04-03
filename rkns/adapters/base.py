from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rkns._zarr import ZarrGroup
from rkns._zarr.storehandler_zarr_v3 import _StoreHandler
from rkns.file_formats import FileFormat

__all__ = ["RKNSBaseAdapter"]


class RKNSBaseAdapter(ABC):
    """
    Abstract base class for RKNS adapters.
    An adapter can import compatible ExG data from other data formats from their raw binary blob into
    the /rkns format.

    Not technically an adapter in an object oriented sense, and more like a Data Converter.
    """

    def __init__(self, handler: _StoreHandler) -> None:
        self._handler = handler
        super().__init__()

    def populate_raw_from_file(
        self, file_path: Path, file_format: FileFormat
    ) -> ZarrGroup:
        # set attributes that are consistent across fileformats here.
        self._handler.raw.attrs["format"] = file_format.value
        return self._populate_raw_from_file(file_path, file_format)

    @abstractmethod
    def _populate_raw_from_file(
        self, file_path: Path, file_format: FileFormat
    ) -> ZarrGroup:
        pass

    @abstractmethod
    def populate_rkns_from_raw(
        self,
        overwrite_if_exists: bool = False,
        validate: bool = True,
    ) -> ZarrGroup:
        pass


# class RKNSIdentityAdapter(RKNSBaseAdapter):
#     """
#     Identity adapter to be used for raw data that is already in the RKNS
#     format, i.e. if the _raw group node has a child group "rkns".

#     """

#     def populate_rkns_from_raw(
#         self,
#         overwrite_if_exists: bool = False,
#         validate: bool = True,
#     ) -> ZarrGroup:
#         try:
#             _raw_rkns = self._handler.raw
#         except KeyError as e:
#             raise ValueError(
#                 f"Group {Names.rkns_root=} does not exist in the Zarr store."
#             ) from e

#         if validate:
#             try:
#                 check_rkns_validity(_raw_rkns)
#             except Exception as e:
#                 raise ValueError(
#                     f"Subgroup {Names.rkns_root=} is not in a valid format."
#                 ) from e

#         _raw_rkns = cast(ZarrGroup, _raw_rkns)

#         new_rkns_node = self._handler.create_group(path=Names.rkns_root)
#         zarr.copy_all(_raw_rkns, new_rkns_node)
#         return new_rkns_node
