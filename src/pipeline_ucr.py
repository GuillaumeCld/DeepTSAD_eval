from eval import Evaluator
from training import Trainer
import pandas as pd
import os
from types import SimpleNamespace

from models import Linear, TimesNet, DLinear, iTransformer, Transformer, TimeMixer, AutoEncoder
from tqdm import tqdm

import numpy as np
import torch
import random



def main():

    # fix seed for reproducibility

    path = 'Datasets/UCR/'
    # construct filelist as all files in UCR
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.endswith('.txt')]

    win_size = 96
    
    # DLinear
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
    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     top_k=3,
    #     d_model=8,
    #     d_ff=16,
    #     num_kernels=6, # number of kernels in InceptionBlock
    #     e_layers=1,    # number of TimesNet blocks
    #     embed='timeF',
    #     freq='t',
    #     dropout=0.1,   # dropout rate
    #     enc_in=1,      # univariate input
    #     c_out=1,       # univariate output
    # )

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
    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     d_model=16,
    #     d_ff=32,
    #     e_layers=2,    # number of TimesNet blocks
    #     enc_in=1,      # univariate input
    #     dec_in=1,      # univariate input
    #     c_out=1,       # univariate output
    #     n_heads=2,
    #     activation='gelu',
    #     embed="fixed",
    #     freq='t',
    #     dropout=0.1,   # dropout rate
    #     channel_independence=True,

    # )

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
    
    # AutoEncoder
    hidden_ratios = [0.5, 0.25]
    config = SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=win_size,
        enc_in=1,
        hidden_ratios=hidden_ratios,
        activation="relu",
    )

    # trainer = Trainer(
    #     batch_size=1024,
    #     lr=1e-2,
    #     device='cuda',
    #     win_size=win_size,
    #     validation_size=0.2,
    #     lr_scheduler=None
    # )
    
    scheduled_lr_scheduler = "plateau"
    scheduled_lr_scheduler_kwargs = {
        "patience": 5,
        "factor": 0.5,
        "min_lr": 1e-5,
    }

    trainer = Trainer(
        batch_size=1024,
        lr=1e-2,
        device='cuda',
        win_size=win_size,
        validation_size=0.2,
        lr_scheduler=scheduled_lr_scheduler,
        lr_scheduler_kwargs=scheduled_lr_scheduler_kwargs
    )
    strategies = ['overlapping']
    for seed in range(3, 4, 1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        results = []
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
        evaluator = Evaluator(batch_size=1024, device='cuda',
                        strategy="overlapping")
        for filename in tqdm(file_list[::20]):

            meta = filename.split('.')[0].split('_')
            split, start, end = int(meta[-3]), int(meta[-2]), int(meta[-1])

            data = np.loadtxt(os.path.join(path, filename))

            data = (data - data.mean()) / data.std()

            train_data = data[:split].reshape(-1, 1)
            test_data = data[split:].reshape(-1, 1)
            anomaly_length = max(end - start + 1, 100)
            start -= split
            end -= split

            model = AutoEncoder.Model(config)

            trainer.train(model, train_data, 100)


            reconstruction = evaluator.reconstruction_error(
                test_data, model, win_size)



            top_ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            sorted_idx = np.argsort(reconstruction)[::-1]

            row = {'filename': filename}
            for k in top_ks:
                topk_idx = sorted_idx[:k]
                hit = int(np.any(
                    (topk_idx >= start - win_size) &
                    (topk_idx <= end + win_size)
                ))
                row[f'score_top{k}'] = hit
                counts[k] += hit

            results.append(row)

            results_df = pd.DataFrame(results)
            results_df.to_csv(
                f'results/AutoEncoder/paper_run_strict_{seed}.csv',
                index=False
            )
        print(f"Seed {seed} - " + ", ".join([f"Top{k}: {counts[k]/len(file_list)*100:.2f}%" for k in top_ks]))
if __name__ == '__main__':
    main()
