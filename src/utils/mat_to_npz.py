import os
import numpy as np
from scipy.io import loadmat

base_path = 'DATASET'
def collect_and_save_data (raw_data_path, collected_data_path):
    X=[]
    y_class = []
    y_dist = []

    for fault_class in range(0,12):
        folder_path = os.path.join(raw_data_path, str(fault_class))
        print(f'processing folder: {folder_path}...')
        if not os.path.exists(folder_path):
            print (f'could not found folder: {fault_class}')
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith('.mat'):
                file_path = os.path.join(folder_path, filename)

                mat_data = loadmat(file_path)

                X.append(mat_data['signals'])
                y_class.append(mat_data['fault'][0][0])
                y_dist.append(mat_data['distance'][0][0])

    X = np.array(X)
    y_class = np.array(y_class)
    y_dist = np.array(y_dist)

    os.makedirs(os.path.dirname(collected_data_path), exist_ok=True)

    np.savez(collected_data_path , signals_VI = X, fault_class = y_class, fault_dist = y_dist)

    print('save complete!')