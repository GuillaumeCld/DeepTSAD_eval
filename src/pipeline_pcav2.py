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
import scipy
import math
from tools import read_file, find_length_rank    

from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.preprocessing import StandardScaler


def main():

    # fix seed for reproducibility

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = 'Datasets/TSB-AD-U/'
    file_list = 'Datasets/File_List/TSB-AD-U-Eva-Full.csv'
    file_list = pd.read_csv(file_list)['file_name'].values


    evaluator = Evaluator(
        metrics='restr',
    )

    all_results = []

    results = []
    for filename in tqdm(file_list):

        _, data, labels = read_file(path, filename)
        rank = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

        window_length = rank


        sequences = [data[i:i+window_length, 0] for i in range(len(data) - window_length + 1)]
        sequences = np.array(sequences)

        # center data and scale by std
        scaler = StandardScaler()
        sequences = scaler.fit_transform(sequences)

        pca = 


        U, S, V = np.linalg.svd(sequences, full_matrices=False)

        # divide each row of U by the corresponding singular value
        weight = S / np.sum(S)

        # compute the weighted distance between each sequence and each singular vector
        distance_matrix = scipy.spatial.distance.cdist(sequences, V, metric='euclidean') / weight


        score = np.sum(distance_matrix, axis=1) 

        if score.shape[0] < sequences.shape[0]:
            score = np.array([score[0]]*math.ceil((window_length-1)/2) + 
                        list(score) + [score[-1]]*((window_length-1)//2))
        metrics = evaluator.metrics_fnc(
            score, labels, slidingWindow=rank)

        result = {'filename': filename}
        result.update(metrics)
        results.append(result)
    
        results_df = pd.DataFrame(results)
        all_results.append(results_df)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv('results/PCA/v2.csv', index=False)


    print(combined_df.mean(numeric_only=True).round(2))


if __name__ == '__main__':
    main()