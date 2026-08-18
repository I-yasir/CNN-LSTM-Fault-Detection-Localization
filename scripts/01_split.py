import yaml
import numpy as np
import h5py
import os

def main():
    print("--- Step 1: Splitting Dataset ---")
    with open("configuration/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(os.path.dirname(config['paths']['splits']), exist_ok=True)
    
    with h5py.File(config['paths']['raw_data'], 'r') as f:
        total_samples = f['signals'].shape[0]
        
    indices = np.arange(total_samples)
    np.random.seed(config['project']['seed'])
    np.random.shuffle(indices)
    
    train_size = int(config['training']['train_ratio'] * total_samples)
    val_size = int(config['training']['val_ratio'] * total_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    np.savez(config['paths']['splits'], 
             train=train_indices, 
             val=val_indices, 
             test=test_indices)
    
    print(f"Splits saved to {config['paths']['splits']}.")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}\n")

if __name__ == "__main__":
    main()