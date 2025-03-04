import hashlib
import os
import tempfile
from pathlib import Path

import pytest
import zarr

from rkns.rkns import RKNS

# paths = ["tests/files/test.edf"]
paths = ["data_shhs1/shhs1-200001.edf"]


def get_file_md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        ref_md5 = hashlib.md5(f.read()).hexdigest()  # type: ignore
    return ref_md5


@pytest.mark.parametrize("path", paths)
def test_populate_rkns_from_raw_edf(path):
    rkns_obj = RKNS.from_file(path, populate_from_raw=True)


# TODO: This will fail, but is a non-trivial problem of how zarr works.
# @pytest.mark.parametrize("path", paths)
# def test_raw_readonly(path):
#     rkns_obj = RKNS.from_file(path, populate_from_raw=False)
#     raw_node = rkns_obj._get_raw()
#     assert not raw_node.store.supports_writes


if __name__ == "main":
    pytest.main()
