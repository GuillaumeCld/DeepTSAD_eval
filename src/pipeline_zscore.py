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
from numba import jit

@jit(nopython=True)
def moving_zscore(x, window=50, threshold=3.0):
    """
    Compute a moving z-score anomaly baseline.
    
    Parameters
    ----------
    x : array-like
        1D time series.
    window : int
        Rolling window size for mean/std estimation.
    threshold : float
        Z-score threshold; |z| > threshold means anomaly.

    Returns
    -------
    z : np.ndarray
        The z-score at each time step.
    anomalies : np.ndarray (bool)
        Boolean mask of detected anomalies.
    """
    x = np.asarray(x)
    n = len(x)

    # rolling mean
    mean = np.zeros(n)
    std = np.zeros(n)

    # simple loop (fast enough for even very large series)
    for t in range(n):
        start = max(0, t - window + 1)
        w = x[start:t+1]
        mean[t] = np.mean(w)
        std[t] = np.std(w) + 1e-8  # avoid division by zero

    z = (x - mean) / std
    anomalies = np.abs(z) > threshold

    return z, anomalies

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

    win_size = 128

    evaluator = Evaluator(
        metrics='restr',
    )

    results = []
    for filename in tqdm(file_list):

        _, data, labels = read_file(path, filename)
        rank = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        score, lab = moving_zscore(data[:, 0])


        if len(score) < len(labels):
            score = np.array([score[0]]*math.ceil((win_size-1)/2) + 
                        list(score) + [score[-1]]*((win_size-1)//2))
            

        metrics = evaluator.metrics_fnc(
            lab, labels, slidingWindow=rank)




        result = {'filename': filename}
        result.update(metrics)
        results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/Zscore/512.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)



if __name__ == '__main__':
    main()