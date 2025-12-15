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
from numba import cuda

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

    all_gpu_devices = [device.id for device in cuda.list_devices()]  # Get a list of all available GPU devices


    results = []
    for filename in tqdm(file_list):

        _, data, labels = read_file(path, filename)
        rank = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        win_size = rank

        matrix_profile = stumpy.stump(data[:, 0], m=win_size)[:, 0]

        matrix_profile = np.array([matrix_profile[0]]*math.ceil((win_size-1)/2) + 
                    list(matrix_profile) + [matrix_profile[-1]]*((win_size-1)//2))
        

        metrics = evaluator.metrics_fnc(
            matrix_profile, labels, slidingWindow=rank)




        result = {'filename': filename}
        result.update(metrics)
        results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/MatrixProfile/64.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)



if __name__ == '__main__':
    main()