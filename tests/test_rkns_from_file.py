import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pyedflib
import pytest
import zarr

from rkns.rkns import RKNS
from rkns.util import check_validity, deep_compare_groups

paths = ["tests/files/test_file.edf"]
# paths = ["data_shhs1/shhs1-200001.edf"]


def get_file_md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()  # type: ignore
    return md5


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_properties(path):
    rkns_obj = RKNS.from_file(path, populate_from_raw=True)
    check_validity(rkns_obj._root)

    fg_names = rkns_obj.get_frequency_group_names()

    with pyedflib.EdfReader(path) as pyedf:
        assert pyedf.getBirthdate() == rkns_obj.patient_info["birthdate"]  # type: ignore
        assert pyedf.getSex() == rkns_obj.patient_info["sex"]  # type: ignore
        # assert pyedf.getSampleFrequencies() == rkns_obj.patientinfo["sex"]  # type: ignore
        pminmax_dminmax = rkns_obj._pminmax_dminmax_by_fg(
            frequency_group=rkns_obj.get_frequency_group_names()[0]
        )
        np.testing.assert_allclose(pyedf.getPhysicalMinimum(), pminmax_dminmax[0])
        np.testing.assert_allclose(pyedf.getPhysicalMaximum(), pminmax_dminmax[1])
        np.testing.assert_allclose(pyedf.getDigitalMinimum(), pminmax_dminmax[2])
        np.testing.assert_allclose(pyedf.getDigitalMaximum(), pminmax_dminmax[3])


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_signals(path):
    rkns_obj = RKNS.from_file(path, populate_from_raw=True)
    check_validity(rkns_obj._root)
    with pyedflib.EdfReader(path) as pyedf:
        pmin = pyedf.getPhysicalMinimum()[:, np.newaxis]
        pmax = pyedf.getPhysicalMaximum()[:, np.newaxis]
        dmin = pyedf.getDigitalMinimum()[:, np.newaxis]
        dmax = pyedf.getDigitalMaximum()[:, np.newaxis]

    channel_data_dig, signal_headers, header = pyedflib.highlevel.read_edf(
        path, digital=True
    )  # type: ignore
    fg = rkns_obj.get_frequency_group_names()[0]
    np.testing.assert_allclose(
        np.array(channel_data_dig).T, rkns_obj._get_digital_signal_by_fg(fg)
    )

    channel_data_phys = pyedflib.highlevel.dig2phys(
        channel_data_dig, pmax=pmax, pmin=pmin, dmin=dmin, dmax=dmax
    )
    phys_rkns = rkns_obj.get_signal_by_fg(fg)
    np.testing.assert_allclose(phys_rkns, channel_data_phys.T)
    # rkns_obj._get_digital_signal_by_fg()


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
        rkns_obj1 = RKNS.from_file(path, populate_from_raw=True)
        rkns_obj1.export(Path(temp_file1))
        reloaded1 = RKNS.from_file(temp_file1)
        assert reloaded1.is_equal_to(rkns_obj1)


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
#     raw_node = rkns_obj._raw
#     assert not raw_node.store.supports_writes


if __name__ == "main":
    pytest.main()
