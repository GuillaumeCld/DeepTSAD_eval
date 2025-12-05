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

from procedure import train_and_evaluate, compare_reconstruction

import pairedtest


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

    win_size = 32

    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
        label_len=win_size,  # unused
        pred_len=0,   # no forecasting for reconstruction
        top_k=3,
        d_model=8,
        d_ff=16,       
        num_kernels=6, # number of kernels in InceptionBlock
        e_layers=1,    # number of TimesNet blocks
        embed='timeF',  
        freq='t',       
        dropout=0.1,   # dropout rate
        enc_in=1,      # univariate input
        c_out=1,       # univariate output 
    )
    

    
    trainer = Trainer(
        batch_size=1024,
        lr=1e-2,
        device='cuda',
        win_size=win_size,
        validation_size=0.2
    )
    evaluator = Evaluator(
        batch_size=1024,
        device='cuda',
        metrics='restr',
        strategy='MSE'
    )
    results = []
    for filename in tqdm(file_list):
        model = TimesNet.Model(config)
        metrics = train_and_evaluate(
            path,
            filename,
            model,
            trainer,
            evaluator,
            win_size=win_size,
            epochs=20
        )
        result = {'filename': filename}
        result.update(metrics)
        results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/TimesNet/MSE.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)


    # filename = file_list[300]
    # evaluator2 = Evaluator(
    #     batch_size=1024,
    #     device='cuda',
    #     strategy='disjoint'
    # )
    # n1, n2 = compare_reconstruction(
    #     path,
    #     filename,
    #     TimesNet.Model(config),
    #     trainer,
    #     evaluator,
    #     evaluator2,
    #     win_size=win_size,
    #     epochs=20
    # )
    # print("P-val of paired distance: " + str(pairedtest.pairedtestpval(n1,n2, 10000)))
    # print(pairedtest.pairedtestconf(n1,n2, 10000))
if __name__ == '__main__':
    main()