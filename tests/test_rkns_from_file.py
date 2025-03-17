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
def pyedf_physical(request):
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
def test_rkns_from_edf_attributes(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital
    reference_fgs = {
        s["label"]: f"fg_{np.round(s['sample_frequency'], 1)}" for s in signal_headers
    }

    fg_names = rkns_obj._get_frequencygroups()
    with pyedflib.EdfReader(path) as pyedf:
        assert pyedf.getBirthdate() == rkns_obj.patient_info["birthdate"]  # type: ignore
        assert pyedf.getSex() == rkns_obj.patient_info["sex"]  # type: ignore
        assert pyedf.getFileDuration() == rkns_obj.get_duration()


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_minmax(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    reference_fgs = dict()
    for s in signal_headers:
        channel_name = s["label"]
        fg = f"fg_{np.round(s['sample_frequency'], 1)}"
        ref_pminmax_dminmax = (
            s["physical_min"],
            s["physical_max"],
            s["digital_min"],
            s["digital_max"],
        )
        reference_fgs[channel_name] = np.array(ref_pminmax_dminmax)

    fgs = rkns_obj._get_frequencygroups()
    for fg in fgs:
        channel_names = rkns_obj._get_channel_names_by_fg(fg)
        rkns_pminmax_dminmax = rkns_obj._pminmax_dminmax_by_fg(fg)
        for i, channel_name in enumerate(channel_names):
            val1 = reference_fgs[channel_name]
            val2 = rkns_pminmax_dminmax[:].T[i]
            np.testing.assert_allclose(val1, val2)


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_digital(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    reference_fgs = dict()
    for s, data in zip(signal_headers, channel_data_dig):
        channel_name = s["label"]
        fg = f"fg_{np.round(s['sample_frequency'], 1)}"
        reference_fgs[channel_name] = np.array(data)

    fgs = rkns_obj._get_frequencygroups()
    for fg in fgs:
        channel_names = rkns_obj._get_channel_names_by_fg(fg)
        rkns_digital_signal = rkns_obj._get_digital_signal_by_fg(fg)
        for i, channel_name in enumerate(channel_names):
            val1 = reference_fgs[channel_name]
            val2 = rkns_digital_signal[:].T[i]
            np.testing.assert_allclose(val1, val2)


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_physical(path, rkns_obj, pyedf_physical):
    channel_data_dig, signal_headers, header = pyedf_physical

    reference_fgs = dict()
    for s, data in zip(signal_headers, channel_data_dig):
        channel_name = s["label"]
        fg = f"fg_{np.round(s['sample_frequency'], 1)}"
        reference_fgs[channel_name] = np.array(data)

    fgs = rkns_obj._get_frequencygroups()
    for fg in fgs:
        channel_names = rkns_obj._get_channel_names_by_fg(fg)
        rkns_physical_signal = rkns_obj._get_signal_by_fg(fg)
        for i, channel_name in enumerate(channel_names):
            val1 = reference_fgs[channel_name]
            val2 = rkns_physical_signal[:].T[i]
            np.testing.assert_allclose(val1, val2)


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
