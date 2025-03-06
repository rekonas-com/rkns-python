from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr
import zarr.codecs
from zarr.codecs.blosc import BloscCname, BloscCodec, BloscShuffle
from zarr.storage import MemoryStore

from rkns.util import copy_attributes, copy_group_recursive, import_from_string


@pytest.fixture
def temp_zarr_store():
    """Create a temporary in-memory zarr store for testing."""
    store = MemoryStore()
    root = zarr.open_group(store=store)
    return root


@pytest.fixture
def source_array(temp_zarr_store):
    """Create a source array with attributes for testing."""
    array = temp_zarr_store.create_array(
        "test_array", shape=(5, 5), dtype=np.float64, fill_value=1.0
    )
    array[:] = np.ones((5, 5))
    array.attrs["key1"] = "value1"
    array.attrs["key2"] = 42
    return array


@pytest.fixture
def source_group(temp_zarr_store):
    """Create a source group with arrays, subgroups and attributes for testing."""
    group = temp_zarr_store.create_group("test_group")
    group.attrs["group_attr1"] = "group_value1"
    group.attrs["group_attr2"] = [1, 2, 3]

    # Add arrays to the group
    array1 = group.create_array("array1", shape=(3, 3), dtype=np.float64)
    array1[:] = np.ones((3, 3))
    array1.attrs["array1_attr"] = "array1_value"

    array2 = group.create_array("array2", shape=(2, 4), dtype=np.float64)
    array2[:] = np.zeros((2, 4))
    array2.attrs["array2_attr"] = "array2_value"

    # Add a subgroup with its own arrays
    subgroup = group.create_group("subgroup")
    subgroup.attrs["subgroup_attr"] = "subgroup_value"

    subarray = subgroup.create_array("subarray", shape=(2, 2), dtype=np.float64)
    subarray[:] = np.ones((2, 2))
    subarray.attrs["subarray_attr"] = "subarray_value"

    return group


def test_copy_attributes_array(temp_zarr_store, source_array):
    """Test copying attributes from one array to another."""
    # Create target array
    target_array = temp_zarr_store.create_array(
        "target_array", shape=(3, 3), dtype=np.float64
    )

    # Copy attributes
    copy_attributes(source_array, target_array)

    # Verify attributes were copied
    assert target_array.attrs["key1"] == "value1"
    assert target_array.attrs["key2"] == 42
    assert len(target_array.attrs) == 2


def test_copy_attributes_group(temp_zarr_store, source_group):
    """Test copying attributes from one group to another."""
    # Create target group
    target_group = temp_zarr_store.create_group("target_group")

    # Copy attributes
    copy_attributes(source_group, target_group)

    # Verify attributes were copied
    assert target_group.attrs["group_attr1"] == "group_value1"
    assert target_group.attrs["group_attr2"] == [1, 2, 3]
    assert len(target_group.attrs) == 2


def test_copy_group_recursive(temp_zarr_store, source_group):
    """Test recursively copying a group and all its contents."""
    # Create target group
    target_group = temp_zarr_store.create_group("target_group")

    # Copy group recursively
    copy_group_recursive(source_group, target_group)
    # Verify group attributes were copied
    assert target_group.attrs["group_attr1"] == "group_value1"
    assert target_group.attrs["group_attr2"] == [1, 2, 3]

    # Verify arrays were copied with their data and attributes
    assert "array1" in target_group
    assert "array2" in target_group
    np.testing.assert_array_equal(target_group["array1"][:], np.ones((3, 3)))
    np.testing.assert_array_equal(target_group["array2"][:], np.zeros((2, 4)))
    assert target_group["array1"].attrs["array1_attr"] == "array1_value"
    assert target_group["array2"].attrs["array2_attr"] == "array2_value"

    # Verify subgroup was copied recursively
    assert "subgroup" in target_group
    assert target_group["subgroup"].attrs["subgroup_attr"] == "subgroup_value"

    # Verify subarray in subgroup
    assert "subarray" in target_group["subgroup"]
    np.testing.assert_array_equal(
        target_group["subgroup"]["subarray"][:], np.ones((2, 2))
    )
    assert (
        target_group["subgroup"]["subarray"].attrs["subarray_attr"] == "subarray_value"
    )


def test_copy_group_recursive_empty_group(temp_zarr_store):
    """Test recursively copying a group with attributes but no children arrays."""
    # Create empty source group
    source_group = temp_zarr_store.create_group("empty_source")
    source_group.attrs["empty"] = True

    target_group = temp_zarr_store.create_group("empty_target")

    copy_group_recursive(source_group, target_group)

    # Verify attributes were copied but no arrays or groups
    assert target_group.attrs["empty"] is True
    assert len(list(target_group.arrays())) == 0
    assert len(list(target_group.groups())) == 0


def test_copy_group_recursive_array_properties(temp_zarr_store):
    """Test that array properties like chunks, compressors and fill_value are preserved."""
    # Create source group with customized array
    source_group = temp_zarr_store.create_group("source")

    # In zarr v3, compressors are specified differently

    compressor = BloscCodec(
        cname=BloscCname.zstd,  # Use enum instead of string
        clevel=3,  # This is fine as int
        shuffle=BloscShuffle.bitshuffle,  # Use enum instead of int 2
        typesize=None,  # This is optional but can be specified if known
    )
    array = source_group.create_array(
        name="custom_array",
        shape=(10, 10),
        dtype=np.float64,
        chunks=(5, 5),
        compressors=compressor,  # In v3, it's compressors (plural)
        fill_value=999.0,
    )
    array[:] = np.ones((10, 10))

    target_group = temp_zarr_store.create_group("target")

    copy_group_recursive(source_group, target_group)

    # Verify array properties were preserved
    target_array = target_group["custom_array"]
    assert target_array.chunks == (5, 5)
    assert target_array.fill_value == 999.0

    # Check compressor
    assert isinstance(target_array.compressors[0], zarr.codecs.BloscCodec)
    assert target_array.compressors[0].cname.value == "zstd"
    assert target_array.compressors[0].clevel == 3
    assert target_array.compressors[0].shuffle == BloscShuffle.bitshuffle


def test_import_from_string():
    """
    Tests that import_from_string correctly imports objects from strings.

    Test translated to pytest from
    https://github.com/django/django/blob/stable/5.1.x/tests/utils_tests/test_module_loading.py

    """

    cls = import_from_string("rkns.util.import_from_string")
    assert cls == import_from_string

    with pytest.raises(ImportError):
        import_from_string("no_dots_in_path")

    with pytest.raises(ModuleNotFoundError):
        import_from_string("utils_tests.unexistent")
