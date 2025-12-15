import torch
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import os
import pandas as pd

def read_file(path, filename):
    file_path = os.path.join(path, filename)

    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    # normalize data globally
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_std = np.where(data_std == 0, 1e-8, data_std)  # Avoid division by zero

    data = (data - data_mean) / data_std

    train_index = filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    return data_train, data, label



class ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size] for i in range(self.sample_num)])
            X = X.unsqueeze(-1)
        else:
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index]


# class ReconstructDataset(torch.utils.data.Dataset):
#     def __init__(self, data, window_size, stride=1, normalize=True):
#         """
#         data   : np.ndarray shape (T,) or (T,1), float-like
#         """
#         super().__init__()
#         data = np.asarray(data, dtype=np.float32)
#         if data.ndim == 1:
#             data = data[:, None]              # (T,1)
#         elif data.shape[1] != 1:
#             raise ValueError(
#                 "Expected univariate data with shape (T,) or (T,1).")

#         self.window_size = int(window_size)
#         self.stride = int(stride)

#         if normalize:
#             mu = data.mean(axis=0, keepdims=True)
#             sigma = data.std(axis=0, keepdims=True)
#             sigma[sigma == 0.0] = 1e-8
#             data = (data - mu) / sigma
#         else:
#             mu = 0.0
#             sigma = 1.0
#         self.data = data                      # (T,1) float32
#         self.mu = mu
#         self.sigma = sigma

#         T = self.data.shape[0]
#         self.sample_num = max(0, (T - self.window_size) // self.stride + 1)

#         # prebuild tensors (fast indexing later)
#         x = torch.from_numpy(self.data)       # (T,1) float32

#         starts = torch.arange(self.sample_num) * self.stride
#         self.X = torch.stack([x[s:s+self.window_size, :]
#                              for s in starts], dim=0)   # (N,W,1)

#     def __len__(self):
#         return self.sample_num

#     def __getitem__(self, idx):

#         return self.X[idx]


def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape)>1: return 0
    if rank==0: return 1
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    

    local_max = argrelextrema(auto_corr, np.greater)[0]

    try:
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]    # Ascending order
        max_local_max = sorted_local_max[0]     # Default
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2: 
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    max_local_max = i 
                    break
        if rank == 3:
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]: 
                    max_local_max = i           
                    break

        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
    

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)