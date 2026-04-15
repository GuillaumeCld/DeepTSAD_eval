from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import Linear, TimesNet, DLinear, iTransformer, Transformer, TimeMixer
from tqdm import tqdm

import numpy as np
import torch
import random
import stumpy
import math
from tools import read_file, find_length_rank
from numba import cuda


import inference

def main():

    # fix seed for reproducibility

    path = 'Datasets/UCR/'
    # construct filelist as all files in UCR
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.endswith('.txt')]

    win_size = 96

    for seed in range(3,8,1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        results = []
        counts = 0

        for filename in tqdm(file_list):

            meta = filename.split('.')[0].split('_')
            split, start, end = int(meta[-3]), int(meta[-2]), int(meta[-1])

            data = np.loadtxt(os.path.join(path, filename)).reshape(-1, 1)

            data = (data - data.mean()) / data.std()

            train_data = data[:split]
            test_data = data[split:]
            anomaly_length = max(end - start + 1, 100)
            start -= split
            end -= split



            sequences_train = np.array([
                train_data[i:i+win_size, 0]
                for i in range(len(train_data) - win_size + 1)
            ])
            train_mean = np.mean(sequences_train, axis=0)
            train_std = np.std(sequences_train, axis=0) + 1e-8
            sequences_train = (sequences_train - train_mean) / train_std
            _, S, Vt = np.linalg.svd(sequences_train, full_matrices=False)
            num_components = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.75) + 1
            V_k = Vt[:num_components]

            sequences = np.array([
                test_data[i:i+win_size, 0]
                for i in range(len(test_data) - win_size + 1)
            ])
            sequences = (sequences - train_mean) / train_std

            projected = sequences @ V_k.T
            reconstructed = projected @ V_k

            errors = (sequences - reconstructed) ** 2
            reconstruction_error = inference.combined_pointwise_profile(errors, len(test_data), win_size)


            anomaly = np.argmax(reconstruction_error)
            # import matplotlib.pyplot as plt
            # plt.plot(reconstruction_error)
            # labels = np.zeros(len(data))
            # labels[start:end+1] = 1
            # plt.scatter(np.where(labels == 1)[0], reconstruction_error[labels == 1], color='red', label='Anomalies')
            # plt.show()

            if start - anomaly_length <= anomaly <= end + anomaly_length:
                results.append({'filename': filename, 'score': 1})
                counts += 1
            else:
                results.append({'filename': filename, 'score': 0})


            results_df = pd.DataFrame(results)
            results_df.to_csv(
                f'results/PCA/hp_ucr_{win_size}.csv', index=False)

        print(
            f'Seed {seed} - Accuracy: {counts}/{len(file_list)} = {counts/len(file_list)*100:.1f}'
        )
if __name__ == '__main__':
    main()
