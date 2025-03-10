import hashlib
import os
import tempfile
from pathlib import Path

import pytest
import zarr

from rkns.rkns import RKNS

paths = ["tests/files/test.edf"]
# paths = ["data_shhs1/shhs1-200001.edf"]


def get_file_md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()  # type: ignore
    return md5


@pytest.mark.parametrize("path", paths)
def test_populate_rkns_from_raw_edf(path):
    rkns_obj = RKNS.from_file(path, populate_from_raw=True)


@pytest.mark.parametrize(
    "path, suffix",
    [(path, suffix) for path in paths for suffix in [".rkns"]],
)
def test_export_filesystem(path, suffix):
    """
    If the file is manually populated, it should not make a difference..
    """

    with (
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        temp_file1 = Path(tmpdir) / f"file1{suffix}"
        # temp_file2 = Path(tmpdir) / "file2.rkns.zip"

        rkns_obj1 = RKNS.from_file(path, populate_from_raw=True)
        # rkns_obj1.populate_rkns_from_raw()

        # rkns_obj2 = RKNS.from_file(path, populate_from_raw=True)

        rkns_obj1.export(Path(temp_file1))
        # rkns_obj2.export(Path(temp_file2))

        reloaded1 = RKNS.from_file(temp_file1)

        tree1 = reloaded1._root.tree()
        tree2 = rkns_obj1._root.tree()
        assert str(tree1) == str(tree2)


@pytest.mark.parametrize(
    "path, suffix",
    [(path, suffix) for path in paths for suffix in [".rkns.zip"]],
)
def test_export_zip(path, suffix):
    with (
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        temp_file1 = Path(tmpdir) / f"file1{suffix}"
        # temp_file2 = Path(tmpdir) / "file2.rkns.zip"

        rkns_obj1 = RKNS.from_file(path, populate_from_raw=True)
        # rkns_obj1.populate_rkns_from_raw()

        # rkns_obj2 = RKNS.from_file(path, populate_from_raw=True)

        with pytest.raises(NotImplementedError):
            rkns_obj1.export(Path(temp_file1))


# TODO: This will fail, but is a non-trivial problem of how zarr works.
# @pytest.mark.parametrize("path", paths)
# def test_raw_readonly(path):
#     rkns_obj = RKNS.from_file(path, populate_from_raw=False)
#     raw_node = rkns_obj._get_raw()
#     assert not raw_node.store.supports_writes


if __name__ == "main":
    pytest.main()
