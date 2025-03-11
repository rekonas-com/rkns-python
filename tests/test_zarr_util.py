from typing import Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import zarr
import zarr.codecs
import zarr.storage
from zarr import AsyncGroup
from zarr.codecs.blosc import BloscCname, BloscCodec, BloscShuffle
from zarr.storage import LocalStore, MemoryStore

from rkns.util.zarr_util import (
    ArrayShapeMismatchError,
    ArrayValueMismatchError,
    AttributeMismatchError,
    GroupComparisonError,
    GroupPathMismatchError,
    MemberCountMismatchError,
    NameMismatchError,
    PathMismatchError,
    add_child_array,
    compare_attrs,
    copy_attributes,
    copy_group_recursive,
    deep_compare_groups,
    get_or_create_target_store,
)


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


class TestCopyAttributesArray:
    def test_copy_attributes_array(self, temp_zarr_store, source_array):
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

    def test_copy_attributes_group(self, temp_zarr_store, source_group):
        """Test copying attributes from one group to another."""
        # Create target group
        target_group = temp_zarr_store.create_group("target_group")

        # Copy attributes
        copy_attributes(source_group, target_group)

        # Verify attributes were copied
        assert target_group.attrs["group_attr1"] == "group_value1"
        assert target_group.attrs["group_attr2"] == [1, 2, 3]
        assert len(target_group.attrs) == 2


class TestCopyGroupRecursive:
    def test_copy_group_recursive(self, temp_zarr_store, source_group):
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
            target_group["subgroup"]["subarray"].attrs["subarray_attr"]
            == "subarray_value"
        )

    def test_copy_group_recursive_empty_group(self, temp_zarr_store):
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

    def test_copy_group_recursive_array_properties(self, temp_zarr_store):
        """Test that array properties like chunks, compressors and fill_value are preserved."""
        source_group = temp_zarr_store.create_group("source")

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


class TestGetTargetStore:
    def test_get_target_store_with_valid_path(self, tmp_path):
        path = tmp_path / "test_store"

        store = get_or_create_target_store(path)

        assert isinstance(store, LocalStore)

        assert store.root == path

    def test_get_target_store_with_existing_path(self, tmp_path):
        path = tmp_path / "existing_store"
        path.mkdir()

        with pytest.raises(
            FileExistsError, match=f"Export target already exists: {path}"
        ):
            get_or_create_target_store(path)

    def test_get_target_store_with_valid_store(self):
        mock_store = MemoryStore()
        store = get_or_create_target_store(mock_store)
        assert store is mock_store

    def test_get_target_store_with_invalid_input(self):
        with pytest.raises(TypeError):
            get_or_create_target_store(123)  # type: ignore


@pytest.fixture
def parent_node(tmp_path):
    store = LocalStore(tmp_path)
    return zarr.group(store=store)


@pytest.fixture
def data():
    return np.random.rand(10, 10)


@pytest.fixture
def name():
    return "test_array"


@pytest.fixture
def attributes():
    return {"key1": "value1", "key2": "value2"}


class TestAddChildArray:
    def test_add_child_array(self, parent_node, data, name, attributes):
        # Call the function
        add_child_array(parent_node, data, name, attributes)

        # Check if the array is created
        assert name in parent_node.array_keys()

        # Retrieve the array
        zarr_array = parent_node[name]

        # Check the shape and dtype
        assert zarr_array.shape == data.shape
        assert zarr_array.dtype == data.dtype

        # Check the data
        np.testing.assert_array_equal(zarr_array[:], data)

        # Check the attributes
        for key, value in attributes.items():
            assert zarr_array.attrs[key] == value

    def test_add_child_array_no_attributes(self, parent_node, data, name):
        # Call the function without attributes
        add_child_array(parent_node, data, name)

        # Check if the array is created
        assert name in parent_node.array_keys()

        # Retrieve the array
        zarr_array = parent_node[name]

        # Check the shape and dtype
        assert zarr_array.shape == data.shape
        assert zarr_array.dtype == data.dtype

        # Check the data
        np.testing.assert_array_equal(zarr_array[:], data)

        # Check that no attributes are set
        assert len(zarr_array.attrs) == 0

    def test_add_child_array_with_compressors(
        self, parent_node, data, name, attributes
    ):
        # Define a compressor

        compressor = BloscCodec(
            cname=BloscCname.zstd,  # Use enum instead of string
            clevel=3,  # This is fine as int
            shuffle=BloscShuffle.bitshuffle,  # Use enum instead of int 2
            typesize=None,  # This is optional but can be specified if known
        )

        # Call the function with compressors
        add_child_array(parent_node, data, name, attributes, compressors=compressor)

        assert name in parent_node.array_keys()

        zarr_array = parent_node[name]

        assert zarr_array.shape == data.shape
        assert zarr_array.dtype == data.dtype

        np.testing.assert_array_equal(zarr_array[:], data)

        for key, value in attributes.items():
            assert zarr_array.attrs[key] == value

        assert zarr_array.compressors[0].cname.value == "zstd"
        assert zarr_array.compressors[0].clevel == 3
        assert zarr_array.compressors[0].shuffle == BloscShuffle.bitshuffle


# Test cases for _compare_attrs function
@pytest.mark.parametrize(
    "attr1, attr2, expected",
    [
        # Test with primitive types
        (5, 5, True),
        (5, 6, False),
        ("hello", "hello", True),
        ("hello", "world", False),
        (3.14, 3.14, True),
        (3.14, 2.71, False),
        (None, None, True),
        (None, 0, False),
        # Test with dictionaries
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, True),
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}, False),
        ({"a": 1, "b": 2}, {"a": 1}, False),
        ({"a": {"b": 2}}, {"a": {"b": 2}}, True),
        ({"a": {"b": 2}}, {"a": {"b": 3}}, False),
        ({"a": "val"}, {"a": "val2"}, False),
        ({"a": "val"}, {"a": "val"}, True),
        # Test with iterables (excluding strings and bytes)
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3], [1, 2, 4], False),
        ((1, 2, 3), (1, 2, 3), True),
        ((1, 2, 3), (1, 2, 4), False),
        ([1, 2, 3], (1, 2, 3), False),
        ([1, 2, 3], (1, 2, 4), False),
        # Test with mixed types
        ({"a": [1, 2, 3], "b": {"c": 4}}, {"a": [1, 2, 3], "b": {"c": 4}}, True),
        ({"a": [1, 2, 3], "b": {"c": 4}}, {"a": [1, 2, 3], "b": {"c": 5}}, False),
    ],
)
def test_compare_attrs(attr1, attr2, expected):
    assert compare_attrs(attr1, attr2) == expected


def test_compare_attrs_empty_structures():
    assert compare_attrs({}, {})
    assert compare_attrs([], [])  # type: ignore
    assert compare_attrs((), ())
    assert not compare_attrs({}, [])  # type: ignore
    assert not compare_attrs([], ())  # type: ignore


def generate_group(path="group1") -> zarr.Group:
    # Create a memory store
    store = zarr.storage.MemoryStore()
    # Create a root group
    group = zarr.group(store=store, path=path)
    return group


def test_deep_compare_async_groups_success():
    group1 = generate_group("group1")
    group2 = generate_group("group1")
    assert deep_compare_groups(group1, group2)


def test_deep_compare_async_groups_name_mismatch():
    mock_group1 = generate_group("group1")
    mock_group2 = generate_group("group2")
    with pytest.raises(NameMismatchError):
        deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_async_groups_member_count_mismatch():
    mock_group1 = generate_group()
    mock_group2 = generate_group()
    mock_group1.create_group("subgroup")

    with pytest.raises(MemberCountMismatchError):
        deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_async_groups_path_mismatch():
    mock_group1 = generate_group()
    mock_group2 = generate_group()
    mock_group1.create_group("subgroup1")
    mock_group2.create_group("subgroup_different")

    with pytest.raises(PathMismatchError):
        deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_async_groups_array_shape_mismatch():
    mock_group1 = generate_group()
    mock_group2 = generate_group()
    arr1 = mock_group1.create_array("array1", dtype=np.float32, shape=(3,))
    arr2 = mock_group2.create_array("array1", dtype=np.float32, shape=(4,))

    with pytest.raises(ArrayShapeMismatchError):
        deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_async_groups_array_shape_match():
    mock_group1 = generate_group()
    mock_group2 = generate_group()
    arr1 = mock_group1.create_array("array1", dtype=np.float32, shape=(4,))
    arr2 = mock_group2.create_array("array1", dtype=np.float32, shape=(4,))

    arr1[:] = np.zeros(4)
    arr2[:] = np.zeros(4)

    deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_async_groups_array_value_mismatch():
    mock_group1 = generate_group()
    mock_group2 = generate_group()
    arr1 = mock_group1.create_array("array1", dtype=np.float32, shape=(4,))
    arr2 = mock_group2.create_array("array1", dtype=np.float32, shape=(4,))

    arr1[:] = np.zeros(4)
    arr2[:] = np.ones(4)

    deep_compare_groups(mock_group1, mock_group2, compare_values=False)
    with pytest.raises(ArrayValueMismatchError):
        deep_compare_groups(mock_group1, mock_group2, compare_values=True)


def test_deep_compare_async_groups_root_name_mismatch():
    mock_group1 = generate_group("a")
    mock_group2 = generate_group("a")
    sg1 = mock_group1.create_group("subgroup1")
    ssg1 = sg1.create_group("grandchild")

    sg2 = mock_group2.create_group("subgroup2")
    ssg2 = sg2.create_group("grandchild")
    with pytest.raises(PathMismatchError):
        deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_async_groups_same_tree():
    mock_group1 = generate_group("a")
    mock_group2 = generate_group("a")
    sg1 = mock_group1.create_group("subgroup1")
    ssg1 = sg1.create_group("grandchild")

    sg2 = mock_group2.create_group("subgroup1")
    ssg2 = sg2.create_group("grandchild")
    deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_async_groups_attribute_mismatch():
    mock_group1 = generate_group("a")
    mock_group2 = generate_group("a")

    mock_group1.attrs["attr1"] = "value1"
    mock_group2.attrs["attr1"] = "value1"
    deep_compare_groups(mock_group1, mock_group2, compare_attributes=True)

    mock_group2.attrs["attr1"] = "value2"
    deep_compare_groups(mock_group1, mock_group2, compare_attributes=False)

    with pytest.raises(AttributeMismatchError):
        deep_compare_groups(mock_group1, mock_group2, compare_attributes=True)


def test_deep_compare_async_groups_array_and_group():
    mock_group1 = generate_group()
    mock_group2 = generate_group()
    arr1 = mock_group1.create_group("abca")
    arr2 = mock_group2.create_array("array1", dtype=np.float32, shape=(4,))

    with pytest.raises(GroupComparisonError):
        deep_compare_groups(mock_group1, mock_group2)


def test_deep_compare_notgroups():
    with pytest.raises(TypeError):
        deep_compare_groups({}, {})  # type: ignore
