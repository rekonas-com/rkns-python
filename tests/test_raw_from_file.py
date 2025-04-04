import hashlib
import tempfile
from pathlib import Path

import pytest

from rkns.rkns import RKNS
from rkns.util import RKNSNodeNames

paths = ["tests/files/test_file.edf"]


def get_file_md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        ref_md5 = hashlib.md5(f.read()).hexdigest()  # type: ignore
    return ref_md5


@pytest.mark.parametrize("path", paths)
def test_raw_md5(path):
    rkns_obj = RKNS.from_file(path, populate_from_raw=False)
    _raw_signal = rkns_obj.handler.raw[RKNSNodeNames.raw_signal.value]
    md5 = _raw_signal.attrs["md5"]  # type: ignore
    ref_md5 = get_file_md5(path)
    assert md5 == ref_md5

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        rkns_obj._reconstruct_original_file(temp_file.name)
        reconstructed_md5 = get_file_md5(temp_file.name)
    assert md5 == ref_md5 == reconstructed_md5


# TODO: This will fail, but is a non-trivial problem of how zarr works.
# @pytest.mark.parametrize("path", paths)
# def test_raw_readonly(path):
#     rkns_obj = RKNS.from_file(path, populate_from_raw=False)
#     raw_node = rkns_obj._get_raw()
#     assert not raw_node.store.supports_writes


if __name__ == "main":
    pytest.main()
