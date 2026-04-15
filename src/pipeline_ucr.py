from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import Linear, TimesNet, DLinear, iTransformer, Transformer, TimeMixer
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

    path = 'Datasets/UCR/'
    # construct filelist as all files in UCR
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.endswith('.txt')]

    win_size = 64
    
    moving_avg = int(win_size * 0.1)
    moving_avg = moving_avg + 1 if moving_avg % 2 == 0 else moving_avg
    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
        label_len=win_size,
        moving_avg=moving_avg,
        dropout=0.1,
        enc_in=1,
        )
    


    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    # )

    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     d_model=8,
    #     d_ff=16,
    #     factor=3,
    #     e_layers=1,    # number of TimesNet blocks
    #     d_layers=1,
    #     enc_in=1,      # univariate input
    #     dec_in=1,      # univariate input
    #     c_out=1,       # univariate output
    #     n_heads=2,
    #     activation='gelu',
    #     moving_avg=25,
    #     embed="fixed",
    #     freq='t',
    #     dropout=0.1,   # dropout rate
    #     down_sampling_window=3,
    #     channel_independence=True,
    #     decomp_method='moving_avg',
    #     down_sampling_layers=2,
    #     use_norm=False,
    #     down_sampling_method="avg"
    # )

    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     d_model=8,
    #     d_ff=16,
    #     factor=3,
    #     e_layers=1,    # number of TimesNet blocks
    #     d_layers=1,
    #     enc_in=1,      # univariate input
    #     dec_in=1,      # univariate input
    #     c_out=1,       # univariate output
    #     n_heads=2,
    #     activation='gelu',
    #     moving_avg=25,
    #     embed="fixed",
    #     freq='t',
    #     dropout=0.1,   # dropout rate
    #     down_sampling_window=3,
    #     channel_independence=True,
    #     decomp_method='moving_avg',
    #     down_sampling_layers=2,
    #     use_norm=False,
    #     down_sampling_method="avg"
    # )

    # timesnet
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

    # dlinear
    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     d_model=8,
    #     d_ff=16,
    #     factor=3,
    #     e_layers=1,    # number of TimesNet blocks
    #     d_layers=1,
    #     enc_in=1,      # univariate input
    #     dec_in=1,      # univariate input
    #     c_out=1,       # univariate output
    #     n_heads=2,
    #     activation='gelu',
    #     moving_avg=25,
    #     embed="fixed",
    #     freq='t',
    #     dropout=0.1,   # dropout rate
    #     down_sampling_window=3,
    #     channel_independence=True,
    #     decomp_method='moving_avg',
    #     down_sampling_layers=2,
    #     use_norm=False,
    #     down_sampling_method="avg"
    # )

    # itransformer
    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     d_model=8,
    #     d_ff=16,
    #     factor=3,
    #     e_layers=1,    # number of TimesNet blocks
    #     d_layers=1,
    #     enc_in=1,      # univariate input
    #     dec_in=1,      # univariate input
    #     c_out=1,       # univariate output
    #     n_heads=2,
    #     activation='gelu',
    #     moving_avg=25,
    #     embed="fixed",
    #     freq='t',
    #     dropout=0.1,   # dropout rate
    #     down_sampling_window=3,
    #     channel_independence=True,
    #     decomp_method='moving_avg',
    #     down_sampling_layers=2,
    #     use_norm=False,
    #     down_sampling_method="avg"
    # )

    # transformer
    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
        label_len=win_size,  # unused
        pred_len=0,   # no forecasting for reconstruction
        d_model=16,
        d_ff=32,
        factor=3,
        e_layers=3,    # number of TimesNet blocks
        d_layers=2,
        enc_in=1,      # univariate input
        dec_in=1,      # univariate input
        c_out=1,       # univariate output
        n_heads=8,
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

    # timemixer
    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     d_model=8,
    #     d_ff=16,   
    #     factor=3,    
    #     e_layers=1,    # number of TimesNet blocks     
    #     d_layers=1,
    #     enc_in=1,      # univariate input
    #     dec_in=1,      # univariate input
    #     c_out=1,       # univariate output 
    #     n_heads=2,
    #     activation='gelu',
    #     moving_avg=25,
    #     embed="fixed",  
    #     freq='t',       
    #     dropout=0.1,   # dropout rate
    #     down_sampling_window=3,
    #     channel_independence=True,
    #     decomp_method='moving_avg',
    #     down_sampling_layers=2,
    #     use_norm=False,
    #     down_sampling_method="avg"
    # )
    

    trainer = Trainer(
        batch_size=1024,
        lr=1e-3,
        device='cuda',
        win_size=win_size,
        validation_size=0.2
    )
    strategies = ['overlapping']
    for seed in range(3, 8, 1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        results = {"disjoint": [], "overlapping": []}
        counts = {"disjoint": 0, "overlapping": 0}
        for filename in tqdm(file_list):

            meta = filename.split('.')[0].split('_')
            split, start, end = int(meta[-3]), int(meta[-2]), int(meta[-1])

            data = np.loadtxt(os.path.join(path, filename))

            data = (data - data.mean()) / data.std()

            train_data = data[:split].reshape(-1, 1)
            test_data = data[split:].reshape(-1, 1)
            anomaly_length = max(end - start + 1, 100)
            start -= split
            end -= split

            model = Transformer.Model(config)
            trainer.train(model, train_data, 20)

            for strat in strategies:
                evaluator = Evaluator(batch_size=1024, device='cuda',
                                      strategy=strat)
                reconstruction = evaluator.reconstruction_error(
                    test_data, model, win_size)

                anomaly = np.argmax(reconstruction)

                if start - anomaly_length <= anomaly <= end + anomaly_length:
                    results[strat].append({'filename': filename, 'score': 1})
                    counts[strat] += 1
                else:
                    results[strat].append({'filename': filename, 'score': 0})


                results_df = pd.DataFrame(results[strat])
                results_df.to_csv(
                    f'results/Transformer/hp_ucr_{win_size}_{strat}_{seed}.csv', index=False)

        for strat in strategies:
            print(
                f'Seed {seed} - {strat} Accuracy: {counts[strat]}/{len(file_list)} = {counts[strat]/len(file_list)*100:.1f}'
            )
if __name__ == '__main__':
    main()
