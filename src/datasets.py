import numpy as np
import h5py
import json
from torch.utils.data import Dataset

class UniversalFaultDataset(Dataset):
    def __init__(self, h5_file, norm_file, indices=None, extra_feature=None):
        self.h5_file = h5_file
        self.indices = indices
        self.extra_feature = extra_feature
        self.file = None 
        
        with open(norm_file, 'r') as f:
            stats = json.load(f)
        self.mean = np.array(stats['mean'], dtype=np.float32)
        self.std = np.array(stats['std'], dtype=np.float32)

        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(indices) if indices is not None else f['signals'].shape[0]

    def _open_hdf5(self):
        if self.file is None:
            self.file = h5py.File(self.h5_file, 'r')
            self.signals = self.file['signals']
            self.classes = self.file['fault_class']
            self.locations = self.file['fault_location']
            if self.extra_feature:
                self.extra = self.file[self.extra_feature]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._open_hdf5()
        real_idx = self.indices[idx] if self.indices is not None else idx
        
        x = np.array(self.signals[real_idx], dtype=np.float32)
        x = (x - self.mean) / (self.std + 1e-8)
        y_cls = np.array(self.classes[real_idx], dtype=np.int64)
        y_loc = np.array(self.locations[real_idx], dtype=np.float32)
        
        if self.extra_feature:
            extra_val = np.array(self.extra[real_idx], dtype=np.float32)
            return x, y_cls, y_loc, extra_val
            
        return x, y_cls, y_loc