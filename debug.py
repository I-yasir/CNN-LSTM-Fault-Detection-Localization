import h5py
import numpy as np

# Replace with your actual dataset path
dataset_path = "data/raw/dataset.h5" 

with h5py.File(dataset_path, "r") as f:
    fault_location = f["fault_location"][:]
    
    print("--- Fault Location Statistics ---")
    print(f"Min value: {np.min(fault_location)}")
    print(f"Max value: {np.max(fault_location)}")
    print(f"Mean value: {np.mean(fault_location)}")
    
    # Check how many values are negative or zero (common culprits)
    below_zero = np.sum(fault_location < 0)
    exact_zero = np.sum(fault_location == 0)
    print(f"Number of negative values: {below_zero}")
    print(f"Number of zero values: {exact_zero}")
    
    # Print the first 10 values just to see the format
    print(f"First 10 values: {fault_location[:10]}")