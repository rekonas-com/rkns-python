import pyedflib
from pathlib import Path
import numpy as np
import pytest

from rkns.RKNS import RKNS

ROOT_DIR = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent

test_file_path = str(Path(TEST_DIR, "test.edf"))

channel_data_physical, channel_headers_physical, header_physical = pyedflib.highlevel.read_edf(
    test_file_path, digital=False
)

channel_data_digital, channel_headers_digital, header_digital = pyedflib.highlevel.read_edf(
    test_file_path, digital=True
)

rkns = RKNS.from_edf(test_file_path, path=str(Path("/tmp/rkns_test/zarr/test.zarr")), overwrite=True)
rkns_load = RKNS.load(path=str(Path("/tmp/rkns_test/zarr/test.zarr")))

def test_write_read_edf_data_array():
    assert np.array_equal(rkns.edf_data_array(), rkns_load.edf_data_array())

def test_edf_values():
    channel_data_rkns_digital = rkns.get_edf_data_records(physical=False)
    for idx in np.arange(len(channel_data_rkns_digital)):
        assert np.array_equal(channel_data_rkns_digital[idx], channel_data_digital[idx])

    channel_data_rkns_physical = rkns.get_edf_data_records()
    for idx in np.arange(len(channel_data_rkns_physical)):
        assert np.array_equal(channel_data_rkns_physical[idx], channel_data_physical[idx])

def test_records_slicing():
    with pytest.raises(IndexError):
        rkns.get_edf_data_records(start=-1)

    num_records = int(rkns.rkns_header["number_of_records"])
    with pytest.raises(IndexError):
        rkns.get_edf_data_records(start=num_records+1)
    
    with pytest.raises(IndexError):
        rkns.get_edf_data_records(start=11, stop=10)

    with pytest.raises(IndexError):
        rkns.get_edf_data_records(start=0, stop=num_records+1)
    
    with pytest.raises(IndexError):
        rkns.get_edf_data_records(start=0, stop=-(num_records+1))

    channel_data_rkns = rkns.get_edf_data_records(start=0, stop=10)
    for idx in np.arange(len(channel_data_rkns)):
        record_size = rkns.rkns_header["record_duration"] * rkns.rkns_header["sample_frequency"][idx]
        assert np.array_equal(channel_data_rkns[idx], channel_data_physical[idx][0:int(10*record_size)])
        assert len(channel_data_rkns[idx]) == 10*record_size

    channel_data_rkns = rkns.get_edf_data_records(start=5, stop=15)
    for idx in np.arange(len(channel_data_rkns)):
        record_size = rkns.rkns_header["record_duration"] * rkns.rkns_header["sample_frequency"][idx]
        assert np.array_equal(channel_data_rkns[idx], channel_data_physical[idx][int(5*record_size):int(15*record_size)])
        assert len(channel_data_rkns[idx]) == 10*record_size

    channel_data_rkns = rkns.get_edf_data_records(start=0, stop=1)
    for idx in np.arange(len(channel_data_rkns)):
        record_size = rkns.rkns_header["record_duration"] * rkns.rkns_header["sample_frequency"][idx]
        assert np.array_equal(channel_data_rkns[idx], channel_data_physical[idx][0:int(1*record_size)])
        assert len(channel_data_rkns[idx]) == 1*record_size

    channel_data_rkns = rkns.get_edf_data_records(start=0, stop=-(num_records//2))
    remaining = num_records - (num_records // 2)
    for idx in np.arange(len(channel_data_rkns)):
        record_size = rkns.rkns_header["record_duration"] * rkns.rkns_header["sample_frequency"][idx]
        assert np.array_equal(channel_data_rkns[idx], channel_data_physical[idx][0:int(remaining*record_size)])
        assert len(channel_data_rkns[idx]) == remaining*record_size

# TODO: Add tests for header files