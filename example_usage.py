from pathlib import Path
import numpy as np
import time

from rkns.RKNS import RKNS


if __name__ == "__main__":

    ROOT_DIR = Path(__file__).parent

    # Provide relative path to EDF and events file from harmonized dataset in BIDS format
    edf_file_path = "" # "data/sub-shhs205804_ses-01_task-sleep_eeg.edf"
    annotations_file_path = "" # "data/sub-shhs205804_ses-01_task-sleep_new_events.tsv"

    if edf_file_path == "" or annotations_file_path == "":
        raise Exception("No data files provided. Please specify the realtive path to a valid EDF file and its corresponding events file.")

    start_time = time.time()

    # Create RKNS object from EDF file
    print("Creating RKNS from EDF file...")
    rkns = RKNS.from_edf(edf_file_path, path=f"zarr/sub-shhs205804_ses-01_task-sleep_eeg.zarr", overwrite=True)    
    
    print(f"--- duration: {time.time() - start_time} seconds ---\n\n")  
    start_time = time.time()
    
    # Add annotations from BIDS events file
    print("Loading annotations from BIDS events file...")
    rkns.load_annotations_from_tsv(annotations_file_path, "stage_AASM_e1", "stage_AASM_e1", np.dtype('S2'), {"name": "stage_AASM_e1"})

    print(f"--- duration: {time.time() - start_time} seconds ---\n\n")

    # Load saved RKNS file
    print("Loading file and preparing samples...")
    rkns = RKNS.load(path="zarr/sub-shhs205804_ses-01_task-sleep_eeg.zarr")

    # Decode annotations
    decode_labels = lambda b: str(b, 'ASCII')
    annotations_array = np.asanyarray(list(map(decode_labels, rkns.annotations_array(name="stage_AASM_e1"))))
    edf_data_array = rkns.get_edf_data_records()

    # Create samples for deeep learning

    epochs = np.asanyarray(np.array_split(edf_data_array[2], annotations_array.shape[0]))
    annotations_array = annotations_array[:,np.newaxis]
    print(f"annotations: {annotations_array.shape}")
    print(f"30s epochs: {epochs.shape}")