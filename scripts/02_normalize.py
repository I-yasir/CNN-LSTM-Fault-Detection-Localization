import yaml
import numpy as np
import h5py
import json

def main():
    print("--- Step 2: Normalization Parameters ---")
    with open("configuration/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    splits = np.load(config['paths']['splits'])
    train_indices = splits['train']
    
    sum_val = np.zeros(config['system']['input_channels'], dtype=np.float64)
    sum_sq = np.zeros(config['system']['input_channels'], dtype=np.float64)
    pixel_count = 0
    batch_size = 1000
    
    with h5py.File(config['paths']['raw_data'], 'r') as f:
        signals_ds = f['signals']
        for i in range(0, len(train_indices), batch_size):
            batch_idx = train_indices[i:i + batch_size]
            sorted_idx = np.sort(batch_idx)
            batch_data = signals_ds[sorted_idx] 
            
            sum_val += np.sum(batch_data, axis=(0, 1))
            sum_sq += np.sum(batch_data ** 2, axis=(0, 1))
            pixel_count += batch_data.shape[0] * batch_data.shape[1]
            
    mean = sum_val / pixel_count
    std = np.sqrt((sum_sq / pixel_count) - (mean ** 2))
    
    with open(config['paths']['norm_params'], 'w') as f:
        json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f, indent=4)
        
    print(f"Saved {config['paths']['norm_params']}.")

if __name__ == "__main__":
    main()