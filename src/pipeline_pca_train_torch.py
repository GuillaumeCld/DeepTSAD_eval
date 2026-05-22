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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Datasets/TSB-AD-U/',
                        help='Path to the dataset directory')
    args = parser.parse_args()


    # fix seed for reproducibility
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = args.data_path
    file_list_path = 'Datasets/File_List/TSB-AD-U-Eva-Full.csv'
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
            num_components = int((energy >= 0.75).nonzero(as_tuple=False)[0][0].item() + 1)
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
            score = inference.combined_pointwise_profile(errors, len(data), window_length)
            # score = inference.disjoint_pointwise_profile(errors, len(data), window_length)

            end_time = time.time()

            metrics = evaluator.metrics_fnc(
                score,
                labels,
                slidingWindow=rank
            )

            result = {'filename': filename, 'execution_time_seconds': end_time - start_time}

            result.update(metrics)
            results.append(result)

        results_df = pd.DataFrame(results)
        all_results.append(results_df)

        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(f'results/evaluation_best/Rec-PCA/seed{seed}_overlapping.csv', index=False)

        print(combined_df.mean(numeric_only=True).round(4))


if __name__ == '__main__':
    main()