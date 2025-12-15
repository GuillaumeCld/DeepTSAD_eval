from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import AnomalyTransformer
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

    path = 'Datasets/UCR/'
    # construct filelist as all files in UCR
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.endswith('.txt')]

    win_size = 32

 
    
    trainer = Trainer(
        batch_size=1024,
        lr=1e-2,
        device='cuda',
        win_size=win_size,
        validation_size=0.2
    )
    evaluator = Evaluator(batch_size=1024, device='cuda',
                          strategy='overlapping')

    count = 0
    results = []

    results = []
    for filename in tqdm(file_list):

        meta = filename.split('.')[0].split('_')
        split, start, end = int(meta[-3]), int(meta[-2]), int(meta[-1])

        data = np.loadtxt(os.path.join(path, filename))

        data = (data - data.mean()) / data.std()

        train_data = data[:split].reshape(-1,1)
        test_data = data[split:].reshape(-1,1)
        anomaly_length = max(end - start + 1, 100)
        start -= split
        end -= split 



        model = AnomalyTransformer.Model(win_size=win_size, enc_in=1, c_out=1)
        trainer.train_anomaly_transformer(model, train_data, 10)

        reconstruction = evaluator.reconstruction_error_anomaly_transformer(
            test_data, model, win_size)
        
        anomaly = np.argmax(reconstruction)

        if start - anomaly_length <= anomaly <= end + anomaly_length:
            results.append({'filename': filename, 'accuracy': 1})
            count += 1
        else:
            results.append({'filename': filename, 'accuracy': 0})

        # metrics = 1 if start - anomaly_length <= anomaly <= end + anomaly_length else 0

        # result = {'filename': filename}
        # result.update(metrics)
        # results.append(result)
        
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(f'results/DLinear/ucr_{win_size}.csv', index=False)
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/AnomalyTransformer/ucr_{win_size}.csv', index=False)
    print(f'Accuracy: {count/len(file_list)*100:.2f}%')



if __name__ == '__main__':
    main()
