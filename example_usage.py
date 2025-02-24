import pyedflib
import numpy as np

from rkns.RKNS import RKNS

if __name__ == "__main__":

    edf_path = "test/test.edf"

    print("Initialize RKNS file")
    rkns = RKNS.create(store="example.rkns", adapter_type_str="rkns.RKNSAdapter.RKNSEdfAdapter", adapter_src_path=edf_path)
    
    # print("Create RKNS from file...")
    # rkns.adapter.from_src(edf_path)

    print("Re-create the EDF from the RKNS file...")
    rkns.adapter.recreate_src(f"out.edf")

    channel_data, channel_headers, header = pyedflib.highlevel.read_edf(
        edf_path, digital=True
    )

    channel_data_rkns, channel_headers_rkns, header_rkns = pyedflib.highlevel.read_edf(
        "out.edf", digital=True
    )

    print("Compare headers...")
    print(header == header_rkns)

    print("Compare channel headers...")
    for a1,a2 in zip(channel_headers, channel_headers_rkns):
        print(np.array_equal(a1,a2))

    print("Check if channel data equal between original an export..")

    for a1,a2 in zip(channel_data, channel_data_rkns):
        print(np.array_equal(a1,a2))
        print(a1)
        print(a2)