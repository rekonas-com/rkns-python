import hashlib
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from rkns.rkns import RKNS
from rkns.util import check_validity
from rkns.util.rkns_util import check_raw_validity, check_rkns_validity, get_freq_group

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
    check_validity(rkns_obj.handler.root)

    check_rkns_validity(rkns_obj.handler.rkns)
    check_raw_validity(rkns_obj.handler.raw)


########### Getter Functions #########
@pytest.mark.parametrize("path", paths)
def test_frequency_groups(path, rkns_obj, pyedf_digital):
    fg_names = rkns_obj._get_frequencygroups()
    channel_data_dig, signal_headers, header = pyedf_digital
    reference_fgs = {get_freq_group(s["sample_frequency"]) for s in signal_headers}

    assert len(fg_names) == len(set(fg_names)), "Frequency groups are not unique.."
    assert set(fg_names) == reference_fgs


@pytest.mark.parametrize("path", paths)
def test_frequency_group(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    reference_fgs = {
        s["label"]: get_freq_group(s["sample_frequency"]) for s in signal_headers
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
        s["label"]: get_freq_group(s["sample_frequency"]) for s in signal_headers
    }

    fg_names = rkns_obj._get_frequencygroups()
    with pyedflib.EdfReader(path) as pyedf:
        assert pyedf.getBirthdate() == rkns_obj.patient_info["birthdate"]  # type: ignore
        assert pyedf.getSex() == rkns_obj.patient_info["sex"]  # type: ignore
        assert pyedf.getFileDuration() == rkns_obj.get_recording_duration()


@pytest.mark.parametrize("path", paths)
def test_rkns_from_edf_minmax(path, rkns_obj, pyedf_digital):
    channel_data_dig, signal_headers, header = pyedf_digital

    reference_fgs = dict()
    for s in signal_headers:
        channel_name = s["label"]
        fg = get_freq_group(s["sample_frequency"])
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
        fg = get_freq_group(s["sample_frequency"])
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
        fg = get_freq_group(s["sample_frequency"])
        reference_fgs[channel_name] = np.array(data)

    fgs = rkns_obj._get_frequencygroups()
    for fg in fgs:
        channel_names = rkns_obj._get_channel_names_by_fg(fg)
        rkns_physical_signal = rkns_obj._get_signal_by_fg(fg)
        rkns_physical_signal_direct = rkns_obj._get_signal_by_freq(float(fg[3:]))

        for i, channel_name in enumerate(channel_names):
            val1 = reference_fgs[channel_name]
            val2 = rkns_physical_signal[:].T[i]
            val3 = rkns_physical_signal_direct[:].T[i]

            np.testing.assert_allclose(val1, val2)
            np.testing.assert_allclose(val2, val3)


@pytest.mark.parametrize("path", paths)
def test_get_signal_by_singlechannel(path, rkns_obj, pyedf_physical):
    """
    For a single channel, test the getter function.
    """
    channel_data_dig, signal_headers, header = pyedf_physical

    reference = defaultdict(dict)
    for s, data in zip(signal_headers, channel_data_dig):
        channel_name = s["label"]
        fg = get_freq_group(s["sample_frequency"])

        reference[channel_name]["data"] = np.array(data)[:, np.newaxis]
        reference[channel_name]["fg"] = fg

    for channel_name in reference.keys():
        signal = rkns_obj.get_signal(channel_name)
        assert signal.shape == reference[channel_name]["data"].shape
        np.testing.assert_allclose(reference[channel_name]["data"], signal)
        assert isinstance(signal, np.ndarray)

    # test with channel not existing
    with pytest.raises(KeyError):
        rkns_obj.get_signal("fail")


@pytest.mark.parametrize("path", paths)
def test_get_signal_by_multichannel_singlegroup(path, rkns_obj, pyedf_physical):
    """
    For multiple channels of a single group, test the getter function.
    The getter should return a single numpy array, with values in order of the channels.
    """
    channel_data_dig, signal_headers, header = pyedf_physical

    # build reference, we need all channels of each group
    reference = defaultdict(dict)
    fg_to_channel = defaultdict(list)
    for s, data in zip(signal_headers, channel_data_dig):
        channel_name = s["label"]
        fg = get_freq_group(s["sample_frequency"])
        reference[channel_name]["data"] = np.array(data)[:, np.newaxis]
        reference[channel_name]["fg"] = fg
        fg_to_channel[fg].append(channel_name)

    fgs = [fg for fg in fg_to_channel.keys()]

    assert len(fgs) >= 3  # sanity check to make sure we have enough fgs for this test

    # test with ALL channels of the group
    channels = fg_to_channel[fgs[0]]
    ref = np.concatenate([reference[c]["data"] for c in channels], 1)
    rkns_signal = rkns_obj.get_signal(channels)
    assert ref.shape == rkns_signal.shape
    np.testing.assert_allclose(ref, rkns_signal)

    # test with subset of channels of the group
    channels = fg_to_channel[fgs[1]][::-2]
    ref = np.concatenate([reference[c]["data"] for c in channels], 1)
    rkns_signal = rkns_obj.get_signal(channels)
    assert ref.shape == rkns_signal.shape
    np.testing.assert_allclose(ref, rkns_signal)

    # test with one channel not existing
    channels = fg_to_channel[fgs[1]][:2]
    with pytest.raises(KeyError):
        rkns_obj.get_signal(channels + ["fail"])

    # test if the channels come from a different fg
    with pytest.raises(ValueError):
        rkns_obj.get_signal(["DC01", "DC04"])


@pytest.mark.parametrize("path", paths)
def test_get_signal_by_frequency(path, rkns_obj, pyedf_physical):
    """
    For a single channel, test the getter function.
    """
    channel_data_dig, signal_headers, header = pyedf_physical

    reference = defaultdict(dict)
    unique_fgs = set()
    for s, data in zip(signal_headers, channel_data_dig):
        channel_name = s["label"]
        fg = get_freq_group(s["sample_frequency"])

        reference[channel_name]["data"] = np.array(data)
        reference[channel_name]["fg"] = fg
        unique_fgs.add((fg, s["sample_frequency"]))

    for fg, sfreq in unique_fgs:
        signal_sfreq = rkns_obj.get_signal(sfreq_Hz=sfreq)
        signal_ref = rkns_obj._get_signal_by_fg(fg)
        assert signal_sfreq.shape == signal_ref.shape

        np.testing.assert_allclose(signal_sfreq, signal_ref)

        channel_names = rkns_obj._get_channel_names_by_fg(fg)
        for i, channel_name in enumerate(channel_names):
            signal_ref_ch = reference[channel_name]["data"]
            signal_rkns_ch = signal_sfreq[:].T[i]

            np.testing.assert_allclose(signal_ref_ch, signal_rkns_ch)
            # np.testing.assert_allclose(val2, val3)

    # test with frequency not existing
    with pytest.raises(KeyError):
        rkns_obj.get_signal(sfreq_Hz=1234567.123)


@pytest.mark.parametrize("path", paths)
def test_get_signal_by_time(path, rkns_obj, pyedf_physical):
    """
    For multiple channels of a single group, test the getter function.
    The getter should return a single numpy array, with values in order of the channels.
    """
    channel_data_dig, signal_headers, header = pyedf_physical

    # build reference, we need all channels of each group
    reference = defaultdict(dict)
    fg_to_channel = defaultdict(list)
    for s, data in zip(signal_headers, channel_data_dig):
        channel_name = s["label"]
        fg = get_freq_group(s["sample_frequency"])
        reference[channel_name]["data"] = np.array(data)[:, np.newaxis]
        reference[channel_name]["fg"] = fg
        fg_to_channel[fg].append(channel_name)

    fgs = [fg for fg in fg_to_channel.keys()]

    assert len(fgs) >= 3  # sanity check to make sure we have enough fgs for this test

    # test with ALL channels of the group
    channels = fg_to_channel[fgs[0]]
    ref = np.concatenate([reference[c]["data"] for c in channels], 1)
    rkns_signal = rkns_obj.get_signal(
        channels,
        time_range=(0, rkns_obj.get_recording_duration() / 2),
    )
    np.testing.assert_allclose(ref[: ref.shape[0] // 2, :], rkns_signal)

    ref = np.concatenate([reference[c]["data"] for c in channels], 1)

    rkns_signal = rkns_obj.get_signal(
        channels[:3],
        time_range=(
            rkns_obj.get_recording_duration() / 4,
            rkns_obj.get_recording_duration() * 3 / 4,
        ),
    )
    np.testing.assert_allclose(
        ref[ref.shape[0] // 4 : ref.shape[0] * 3 // 4, :3], rkns_signal
    )

    with pytest.raises(ValueError):
        rkns_signal = rkns_obj.get_signal(channels[:3], time_range=(1, 0))


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
