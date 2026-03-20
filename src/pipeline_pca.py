from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import TimesNet
from tqdm import tqdm

import numpy as np
import torch
import random
import stumpy
import math
from tools import read_file, find_length_rank    

def main():

    # fix seed for reproducibility

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = 'Datasets/TSB-AD-U/'
    file_list = 'Datasets/File_List/TSB-AD-U-Eva.csv'
    file_list = pd.read_csv(file_list)['file_name'].values


    evaluator = Evaluator(
        metrics='restr',
    )

    all_results = []
    window_length = 64

    results = []
    for filename in tqdm(file_list):

        data_train, data, labels = read_file(path, filename)

        sequences = [data[i:i+window_length, 0] for i in range(len(data) - window_length + 1)]
        sequences = np.array(sequences)
        # center data and scale by std
        sequences = (sequences - np.mean(sequences, axis=0)) / np.std(sequences, axis=0)

        U, S, V = np.linalg.svd(sequences, full_matrices=False)
        # keep 90% of the variance
        variance_explained = np.cumsum(S**2) / np.sum(S**2)
        num_components = np.searchsorted(variance_explained, 0.95) + 1
        # take all components 
        # num_components = len(S) - 1
        reconstructed = U[:, :num_components] @ np.diag(S[:num_components]) @ V[:num_components]


        ## Analytical reconstruction error metrics
        # frobenius norm
        frobenius_norm = np.sum(S[num_components:]**2)
        relative_frobenius_norm = np.sqrt(frobenius_norm / np.sum(S**2))
        # spectral norm
        spectral_norm = S[num_components]
        relative_spectral_norm = spectral_norm / S[0]

        ## Empirical reconstruction error metrics per label
        # normal points
        normal_indices = np.where(labels[:-window_length+1] == 0)[0]
        normal_relative_frobenius_norm = np.sqrt(np.sum((sequences[normal_indices] - reconstructed[normal_indices])**2) / np.sum(sequences[normal_indices]**2))
        normal_relative_spectral_norm = np.max(np.linalg.norm(sequences[normal_indices] - reconstructed[normal_indices], axis=1)) / np.max(np.linalg.norm(sequences[normal_indices], axis=1))
        # anomalous points
        anomalous_indices = np.where(labels[:-window_length+1] == 1)[0]
        anomalous_relative_frobenius_norm = np.sqrt(np.sum((sequences[anomalous_indices] - reconstructed[anomalous_indices])**2) / np.sum(sequences[anomalous_indices]**2))
        anomalous_relative_spectral_norm = np.max(np.linalg.norm(sequences[anomalous_indices] - reconstructed[anomalous_indices], axis=1)) / np.max(np.linalg.norm(sequences[anomalous_indices], axis=1))


        score = np.linalg.norm(sequences - reconstructed, axis=1)

        rank = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)


        metrics = evaluator.metrics_fnc(
            score, labels[:-window_length+1], slidingWindow=rank)

        result = {'filename': filename, 'relative_frobenius_norm': relative_frobenius_norm, 'relative_spectral_norm': relative_spectral_norm,
                    'normal_relative_frobenius_norm': normal_relative_frobenius_norm, 'normal_relative_spectral_norm': normal_relative_spectral_norm,
                    'anomalous_relative_frobenius_norm': anomalous_relative_frobenius_norm, 'anomalous_relative_spectral_norm': anomalous_relative_spectral_norm}
        result.update(metrics)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    all_results.append(results_df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv('results/PCA/64_95.csv', index=False)


    print(combined_df.mean(numeric_only=True).round(4))


if __name__ == '__main__':
    main()