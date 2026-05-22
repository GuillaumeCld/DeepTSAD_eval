from eval import Evaluator
import pandas as pd
import inference
from tqdm import tqdm

import numpy as np
import torch
import random
import time

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
    file_list_path = 'Datasets/File_List/TSB-AD-U-Tuning.csv'
    file_list = pd.read_csv(file_list_path)['file_name'].values

    evaluator = Evaluator(metrics='restr')

    all_results = []
    results = []
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    window_lengths = [32, 64, 96]
    explained_variance_thresholds = [0.25, 0.5, 0.75, 0.95]

    for windows in window_lengths:
        window_length = windows
        for thresholds in explained_variance_thresholds:
            explained_variance_threshold = thresholds

            trial_times = []
            trial_scores = []
            for filename in file_list:

                data_train, data, labels = read_file(path, filename)
                rank = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)



                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                start_time = time.time()


                # ----- TRAIN WINDOWS -----
                sequences_train = torch.stack([
                    torch.from_numpy(data_train[i:i+window_length, 0].astype(np.float32))
                    for i in range(len(data_train) - window_length + 1)
                ]).to(device)

                train_mean = sequences_train.mean(dim=0, keepdim=True)
                train_std = sequences_train.std(dim=0, unbiased=False, keepdim=True) + 1e-8

                sequences_train = (sequences_train - train_mean) / train_std

                # ----- SVD -----

                U, S, Vh = torch.linalg.svd(sequences_train, full_matrices=False)


                energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
                num_components = int((energy >= explained_variance_threshold).nonzero(as_tuple=False)[0][0].item() + 1)
                V_k = Vh[:num_components, :]


                # ----- FULL DATA WINDOWS -----
                sequences = torch.stack([
                    torch.from_numpy(data[i:i+window_length, 0].astype(np.float32))
                    for i in range(len(data) - window_length + 1)
                ]).to(device)

                sequences = (sequences - train_mean) / train_std

                # ----- PCA RECONSTRUCTION -----
                projected = sequences @ V_k.T
                reconstructed = projected @ V_k

                # ----- ANOMALY SCORE -----
                errors = (sequences - reconstructed) ** 2
                errors = errors.cpu().numpy()
                # score = inference.combined_pointwise_profile(errors, len(data), window_length)
                score = inference.disjoint_pointwise_profile(errors, len(data), window_length)

                end_time = time.time()

                metrics = evaluator.metrics_fnc(
                    score,
                    labels,
                    slidingWindow=rank
                )

                trial_times.append(end_time - start_time)
                trial_scores.append(metrics['AUC-PR'])

            print(f"Window Length: {window_length}, Explained Variance Threshold: {explained_variance_threshold}, Average Time: {np.mean(trial_times):.3f} seconds, Average AUC-PR: {np.mean(trial_scores):.2f}")


if __name__ == '__main__':
    main()