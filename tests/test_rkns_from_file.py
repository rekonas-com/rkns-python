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
from rkns.util.rkns_util import check_raw_validity, check_rkns_validity

paths = ["tests/files/test_file.edf"]
# paths = ["data_shhs1/shhs1-200001.edf"]


@pytest.fixture(params=paths)
def rkns_obj(request):
    """
    Fixture for the RKNS object for each path
    """
    path = request.param
    # Create and return the RKNS object
    return RKNS.from_file(path, populate_from_raw=True)


@pytest.fixture(params=paths)
def pyedf_digital(request):
    """
    Fixture for the pyedf output for each path (returning the digital values)
    """
    path = request.param

    # channel_data_dig, signal_headers, header
    return pyedflib.highlevel.read_edf(path, digital=True)  # type: ignore


@pytest.fixture(params=paths)
def pyedf_pysical(request):
    """
    Fixture for the pyedf output for each path (returning the physical values)
    """
    path = request.param

    # channel_data_dig, signal_headers, header
    return pyedflib.highlevel.read_edf(path, digital=False)  # type: ignore


def get_file_md5(path: str | Path) -> str:
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()  # type: ignore
    return md5


@pytest.mark.parametrize("path", paths)
def test_validity(path, rkns_obj):
    check_validity(rkns_obj._root)

    check_rkns_validity(rkns_obj._rkns)
    check_raw_validity(rkns_obj._raw)


########### Getter Functions #########
@pytest.mark.parametrize("path", paths)
def test_frequency_groups(path, rkns_obj, pyedf_digital):
    breakpoint()
    fg_names = rkns_obj._get_frequencygroups()
    channel_data_dig, signal_headers, header = pyedf_digital
    reference_fgs = {f"fg_{np.round(s['sample_frequency'], 1)}" for s in signal_headers}

    assert len(fg_names) == len(set(fg_names)), "Frequency groups are not unique.."
    assert set(fg_names) == reference_fgs


@pytest.mark.parametrize("path", paths)
def test_frequency_group(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    reference_fgs = {
        s["label"]: f"fg_{np.round(s['sample_frequency'], 1)}" for s in signal_headers
    }

    for channel_name, fg_name in reference_fgs.items():
        assert fg_name == rkns_obj._get_frequencygroup(channel_name)


@pytest.mark.parametrize("path", paths)
def test_channel_names(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    reference_channel_names = {s["label"] for s in signal_headers}
    channel_names = rkns_obj.get_channel_names()

    assert len(reference_channel_names) == len(set(channel_names)), (
        "Frequency groups are not unique.."
    )
    assert set(channel_names) == reference_channel_names


@pytest.mark.parametrize("path", paths)
def test_frequency_by_channel(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    for s in signal_headers:
        channel_name = s["label"]
        frequency = s["sample_frequency"]
        assert np.isclose(frequency, rkns_obj.get_frequency_by_channel(channel_name))


###################################################################


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_properties(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    fg_names = rkns_obj._get_frequencygroups()
    with pyedflib.EdfReader(path) as pyedf:
        assert pyedf.getBirthdate() == rkns_obj.patient_info["birthdate"]  # type: ignore
        assert pyedf.getSex() == rkns_obj.patient_info["sex"]  # type: ignore
        # assert pyedf.getSampleFrequencies() == rkns_obj.patientinfo["sex"]  # type: ignore
        pminmax_dminmax = rkns_obj._pminmax_dminmax_by_fg(
            frequency_group=rkns_obj._get_frequencygroup()[0]
        )
        np.testing.assert_allclose(pyedf.getPhysicalMinimum(), pminmax_dminmax[0])
        np.testing.assert_allclose(pyedf.getPhysicalMaximum(), pminmax_dminmax[1])
        np.testing.assert_allclose(pyedf.getDigitalMinimum(), pminmax_dminmax[2])
        np.testing.assert_allclose(pyedf.getDigitalMaximum(), pminmax_dminmax[3])


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_signals(path, rkns_obj):
    with pyedflib.EdfReader(path) as pyedf:
        pmin = pyedf.getPhysicalMinimum()[:, np.newaxis]
        pmax = pyedf.getPhysicalMaximum()[:, np.newaxis]
        dmin = pyedf.getDigitalMinimum()[:, np.newaxis]
        dmax = pyedf.getDigitalMaximum()[:, np.newaxis]

    channel_data_dig, signal_headers, header = pyedflib.highlevel.read_edf(
        path, digital=True
    )  # type: ignore
    fg = rkns_obj._get_frequencygroups()[0]
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
