from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import zarr
import zarr.errors

from ..util import RKNSNodeNames as Names
from ..util import check_rkns_validity

__all__ = ["RKNSBaseAdapter", "RKNSIdentityAdapter"]


class RKNSBaseAdapter(ABC):
    """
    Abstract base class for RKNS adapters.
    An adapter can import compatible ExG data from other data formats from their raw binary blob into
    the /rkns format.

    Not technically an adapter in an object oriented sense, and more like a Data Converter.
    """

    @abstractmethod
    def populate_rkns_from_raw(
        self,
        raw_node: zarr.Group,
        root_node: zarr.Group,
        overwrite_if_exists: bool = False,
        validate: bool = True,
    ) -> zarr.Group:
        """
        Takes as input the zarr raw Group and creates a new child group "rkns" to the provided root_node.
        The created node follows the structure of the "/rkns" subgroup according to the RKNS specifications.

        Parameters
        ----------
        raw_node
            Group node corresponding to "/_raw" in the RKNS specs.
        root_node
            Group node corresponding to "/" in the RKNS specs.
        validate
            Whether to do some validation of the provided raw data,
            by default True.
            Group node corresponding to "/" in the RKNS specs.
        overwrite_if_exists
            If set to True, use write mode "w" for overwriting an existing /rkns child group if it exists.
            If set to False, use write mode "w-", which fails if the child group /rkns exists.

        Returns
        -------
            The created node that was added as a child to the root_node.
        """


class RKNSIdentityAdapter(RKNSBaseAdapter):
    """
    Identity adapter to be used for raw data that is already in the RKNS
    format, i.e. if the _raw group node has a child group "rkns".

    """

    def populate_rkns_from_raw(
        self,
        raw_node: zarr.Group,
        root_node: zarr.Group,
        overwrite_if_exists: bool = False,
        validate: bool = True,
    ) -> zarr.Group:
        try:
            _raw_rkns = raw_node[Names.rkns_root]
        except KeyError as e:
            raise ValueError(
                f"Group {Names.rkns_root=} does not exist in the Zarr store."
            ) from e

        if validate:
            try:
                check_rkns_validity(_raw_rkns)
            except Exception as e:
                raise ValueError(
                    f"Subgroup {Names.rkns_root=} is not in a valid format."
                ) from e

        _raw_rkns = cast(zarr.Group, _raw_rkns)

        mode = "w" if overwrite_if_exists else "w-"
        new_rkns_node = root_node.create_group(name=Names.rkns_root, mode=mode)

        zarr.copy_all(_raw_rkns, new_rkns_node)
        return new_rkns_node
