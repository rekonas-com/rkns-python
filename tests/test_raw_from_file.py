import hashlib
import os
import tempfile
from pathlib import Path

import pytest
import zarr

from rkns.rkns import RKNS

paths = ["tests/files/test.edf"]


def get_file_md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        ref_md5 = hashlib.md5(f.read()).hexdigest()  # type: ignore
    return ref_md5


@pytest.mark.parametrize("path", paths)
def test_raw_md5(path):
    rkns_obj = RKNS.from_file(path, populate_from_raw=False)

    md5 = rkns_obj._get_raw_signal().attrs["md5"]  # type: ignore
    ref_md5 = get_file_md5(path)
    assert md5 == ref_md5

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        rkns_obj._reconstruct_original_file(temp_file.name)
        reconstructed_md5 = get_file_md5(temp_file.name)
    assert md5 == ref_md5 == reconstructed_md5
    tmp = rkns_obj._get_raw_signal()


# TODO: This will fail, but is a non-trivial problem of how zarr works.
# @pytest.mark.parametrize("path", paths)
# def test_raw_readonly(path):
#     rkns_obj = RKNS.from_file(path, populate_from_raw=False)
#     raw_node = rkns_obj._get_raw()
#     assert not raw_node.store.supports_writes


if __name__ == "main":
    pytest.main()
