from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import TimeMixer
from tqdm import tqdm

import numpy as np
import torch
import random

import tools 

from procedure import train_and_evaluate

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
        d_model=8,
        d_ff=16,   
        factor=3,    
        e_layers=1,    # number of TimesNet blocks     
        d_layers=1,
        enc_in=1,      # univariate input
        dec_in=1,      # univariate input
        c_out=1,       # univariate output 
        n_heads=2,
        activation='gelu',
        moving_avg=25,
        embed="fixed",  
        freq='t',       
        dropout=0.1,   # dropout rate
        down_sampling_window=3,
        channel_independence=True,
        decomp_method='moving_avg',
        down_sampling_layers=2,
        use_norm=False,
        down_sampling_method="avg"
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
        strategy='overlapping'
    )
    results = []
    for filename in tqdm(file_list):
        model = TimeMixer.Model(config)
        metrics = train_and_evaluate(
            path,
            filename,
            model,
            trainer,
            evaluator,
            win_size=win_size,
            epochs=50
        )
        _, data, labels = tools.read_file(path, filename)

        relative_error = evaluator.relative_reconstruction_error(
            data, model, win_size)

        rel_erruer_normal = relative_error[labels == 0].mean()
        rel_erreur_anomalie = relative_error[labels == 1].mean()
        top_1_normal = relative_error[labels == 0].max()
        top_1_anomalie = relative_error[labels == 1].max()

        result = {'filename': filename,
                  'rel_normal': rel_erruer_normal.item(),
                  'rel_abnormal': rel_erreur_anomalie.item(),
                  'top1_normal': top_1_normal.item(),
                  'top1_abnormal': top_1_anomalie.item()
                  }
        result.update(metrics)
        results.append(result)
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/TimeMixer/32_50.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)
if __name__ == '__main__':
    main()