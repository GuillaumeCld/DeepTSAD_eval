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
    file_list = 'Datasets/File_List/TSB-AD-U-Eva-Full.csv'
    file_list = pd.read_csv(file_list)['file_name'].values


    evaluator = Evaluator(
        metrics='restr',
    )

    all_results = []
    
    for run in range(50):
        results = []
        for filename in tqdm(file_list, desc=f"Run {run+1}/50"):

            _, data, labels = read_file(path, filename)
            rank = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

            # compute a random score between 0 and 1 with a probability of 0.001 of being an anomaly 
            score = np.random.rand(len(data))
            score[np.random.rand(len(data)) < 0.001] += 1.0

            metrics = evaluator.metrics_fnc(
                score, labels, slidingWindow=rank)

            result = {'filename': filename}
            result.update(metrics)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        all_results.append(results_df)
    
    # Combine all runs and compute average metrics
    combined_df = pd.concat(all_results, ignore_index=True)
    avg_metrics = combined_df.groupby('filename').mean(numeric_only=True).mean().round(3) * 100
    
    combined_df.to_csv('results/Random/001.csv', index=False)
    print("Average metrics across 50 runs:")
    print(avg_metrics)



if __name__ == '__main__':
    main()