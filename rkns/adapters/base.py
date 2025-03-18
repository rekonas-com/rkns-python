from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import run
from typing import cast

import zarr
import zarr.errors
from zarr.core.group import GroupMetadata

from rkns.util import RKNSNodeNames

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

    @classmethod
    @abstractmethod
    def populate_rkns_from_raw(
        cls,
        raw_node: zarr.Group,
        root_node: zarr.Group,
        overwrite_if_exists: bool = False,
        validate: bool = True,
    ) -> zarr.Group:
        pass

    @classmethod
    def create_rkns_group_structure(
        cls, root_node: zarr.Group, overwrite_if_exists: bool
    ) -> tuple[zarr.Group, zarr.Group, zarr.Group]:
        _rkns_name = RKNSNodeNames.rkns_root.value
        _rkns_signals_name = RKNSNodeNames.rkns_signals_group.value
        _rkns_annotations_name = RKNSNodeNames.rkns_annotations_group.value

        try:
            hierarchy = root_node.create_hierarchy(
                {
                    f"{_rkns_name}/{_rkns_signals_name}": GroupMetadata(),
                    f"{_rkns_name}/{_rkns_annotations_name}": GroupMetadata(),
                }
            )
            node_dict = {key: value for key, value in hierarchy}

        except zarr.errors.ContainsGroupError as e:
            if overwrite_if_exists:
                run(root_node.store.delete_dir(_rkns_signals_name))

                hierarchy = root_node.create_hierarchy(
                    {
                        f"{_rkns_name}/{_rkns_signals_name}": GroupMetadata(),
                        f"{_rkns_name}/{_rkns_annotations_name}": GroupMetadata(),
                    }
                )
                node_dict = {key: value for key, value in hierarchy}

            else:
                raise RuntimeError(
                    "The group node /rkns already exists."
                    + "\n For overwriting it, set 'overwrite_if_exists=True'"
                ) from e

        # we only created groups, so we can cast properly..
        # _rkns_signals_node = cast(Iterator[tuple[str, zarr.Group]], _rkns_signals_node)
        return (
            cast(zarr.Group, node_dict[f"{_rkns_name}"]),
            cast(zarr.Group, node_dict[f"{_rkns_name}/{_rkns_signals_name}"]),
            cast(zarr.Group, node_dict[f"{_rkns_name}/{_rkns_annotations_name}"]),
        )


class RKNSIdentityAdapter(RKNSBaseAdapter):
    """
    Identity adapter to be used for raw data that is already in the RKNS
    format, i.e. if the _raw group node has a child group "rkns".

    """

    @classmethod
    def populate_rkns_from_raw(
        cls,
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
