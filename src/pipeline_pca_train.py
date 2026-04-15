from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace
import inference
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
    file_list_path = 'Datasets/File_List/TSB-AD-U-Eva.csv'
    file_list = pd.read_csv(file_list_path)['file_name'].values

    evaluator = Evaluator(metrics='restr')

    all_results = []
    results = []

    for seeds in range(1):
        seed = seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        for filename in tqdm(file_list):

            data_train, data, labels = read_file(path, filename)
            rank = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)


            window_length = 96


            # ----- TRAIN WINDOWS -----
            sequences_train = np.array([
                data_train[i:i+window_length, 0]
                for i in range(len(data_train) - window_length + 1)
            ])

            train_mean = np.mean(sequences_train, axis=0)
            train_std = np.std(sequences_train, axis=0) + 1e-8

            sequences_train = (sequences_train - train_mean) / train_std

            # ----- SVD -----
            _, S, Vt = np.linalg.svd(sequences_train, full_matrices=False)

            # keep components
            num_components = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.75) + 1
            V_k = Vt[:num_components]

            # ----- FULL DATA WINDOWS -----
            sequences = np.array([
                data[i:i+window_length, 0]
                for i in range(len(data) - window_length + 1)
            ])

            sequences = (sequences - train_mean) / train_std

            # ----- PCA RECONSTRUCTION -----
            projected = sequences @ V_k.T
            reconstructed = projected @ V_k

            # ----- ANALYTICAL ERRORS -----
            frobenius_norm = np.sum(S[num_components:] ** 2)
            relative_frobenius_norm = np.sqrt(frobenius_norm / np.sum(S ** 2))

            spectral_norm = S[num_components] if num_components < len(S) else 0
            relative_spectral_norm = spectral_norm / S[0]

            # ----- EMPIRICAL ERRORS -----
            label_windows = labels[:-window_length + 1]

            normal_indices = np.where(label_windows == 0)[0]
            anomalous_indices = np.where(label_windows == 1)[0]

            normal_relative_frobenius_norm = np.sqrt(
                np.sum((sequences[normal_indices] - reconstructed[normal_indices])**2) /
                np.sum(sequences[normal_indices]**2)
            ) if len(normal_indices) > 0 else np.nan

            normal_relative_spectral_norm = (
                np.max(np.linalg.norm(sequences[normal_indices] - reconstructed[normal_indices], axis=1)) /
                np.max(np.linalg.norm(sequences[normal_indices], axis=1))
            ) if len(normal_indices) > 0 else np.nan

            anomalous_relative_frobenius_norm = np.sqrt(
                np.sum((sequences[anomalous_indices] - reconstructed[anomalous_indices])**2) /
                np.sum(sequences[anomalous_indices]**2)
            ) if len(anomalous_indices) > 0 else np.nan

            anomalous_relative_spectral_norm = (
                np.max(np.linalg.norm(sequences[anomalous_indices] - reconstructed[anomalous_indices], axis=1)) /
                np.max(np.linalg.norm(sequences[anomalous_indices], axis=1))
            ) if len(anomalous_indices) > 0 else np.nan

            ratio_normal_abnormal_frobenius = (
                normal_relative_frobenius_norm / anomalous_relative_frobenius_norm
            ) if anomalous_relative_frobenius_norm > 0 else np.nan

            # ----- ANOMALY SCORE -----
            # score = np.linalg.norm(sequences - reconstructed, axis=1)
            errors = (sequences - reconstructed) ** 2
            # score = inference.combined_pointwise_profile(errors, len(data), window_length)
            score = inference.disjoint_pointwise_profile(errors, len(data), window_length)
            # score
            # import matplotlib.pyplot as plt
            # plt.plot(score)
            # plt.scatter(np.where(labels == 1)[0], score[labels == 1], color='red', label='Anomalies')
            # plt.show()
            metrics = evaluator.metrics_fnc(
                score,
                labels,
                slidingWindow=rank
            )

            result = {
                'filename': filename,
                'relative_frobenius_norm': relative_frobenius_norm,
                'relative_spectral_norm': relative_spectral_norm,
                'normal_relative_frobenius_norm': normal_relative_frobenius_norm,
                'normal_relative_spectral_norm': normal_relative_spectral_norm,
                'anomalous_relative_frobenius_norm': anomalous_relative_frobenius_norm,
                'anomalous_relative_spectral_norm': anomalous_relative_spectral_norm
            }

            result.update(metrics)
            results.append(result)

        results_df = pd.DataFrame(results)
        all_results.append(results_df)

        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(f'results/PCA/96_75_disjoint_{seed}.csv', index=False)

        print(combined_df.mean(numeric_only=True).round(4))


if __name__ == '__main__':
    main()