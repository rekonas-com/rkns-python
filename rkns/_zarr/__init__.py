from .generics import ZarrArray, ZarrGroup  # isort: skip (IMPORTANT! )

from .adapter import StoreHandler, zarr_utils
from .utils_interface import TreeRepr

add_child_array = zarr_utils.add_child_array
get_or_create_target_store = zarr_utils.get_or_create_target_store
copy_attributes = zarr_utils.copy_attributes
copy_group_recursive = zarr_utils.copy_group_recursive
deep_compare_groups = zarr_utils.deep_compare_groups
group_tree_with_attrs = zarr_utils.group_tree_with_attrs
get_codec = zarr_utils.get_codec
compare_attrs = zarr_utils.compare_attrs


__all__ = [
    "TreeRepr",
    "ZarrArray",
    "ZarrGroup",
    "add_child_array",
    "get_or_create_target_store",
    "copy_attributes",
    "copy_group_recursive",
    "deep_compare_groups",
    "group_tree_with_attrs",
    "get_codec",
    "compare_attrs",
    "StoreHandler",
]
