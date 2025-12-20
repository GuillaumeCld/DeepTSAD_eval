from eval import Evaluator
from training import Trainer
import pandas as pd
from types import SimpleNamespace

from models import Linear, DLinear, TimesNet, iTransformer, Transformer
from tqdm import tqdm

import numpy as np
import torch
import random
from procedure import train_and_evaluate
import tools
import time

def main():

    # fix seed for reproducibility

    seed = 3
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


    strides = [1]#, 2, 4, 8, 16]

    trainer = Trainer(
        batch_size=1024,
        lr=1e-2,
        device='cuda',
        win_size=win_size,
        validation_size=0.2
    )
    
    strategies = ["overlapping"]

    for seed in range(3):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        results = {"disjoint": [], "overlapping": []}

        for filename in tqdm(file_list):
            model = TimesNet.Model(config)
            
            data_train, data, labels = tools.read_file(path, filename)
            trainer.win_size = win_size
            trainer.train(model, data_train, 50)
            

            for strat in strategies:
                evaluator = Evaluator(
                    batch_size=1024,
                    device='cuda',
                    metrics='restr',
                    strategy=strat
                )
                for stride in strides:
                    start_time = time.time()
                    reconstruction_error = evaluator.reconstruction_error(data, model, win_size, stride)
                    end_time = time.time()

                    rank = tools.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
                    metrics = evaluator.metrics_fnc(
                    reconstruction_error, labels, slidingWindow=rank)

                    result = {'filename': filename, 'inference_time': end_time - start_time, 'stride': stride}
                    result.update(metrics)
                    results[strat].append(result)
                
                    results_df = pd.DataFrame(results[strat])
                    results_df.to_csv(f'results/TimesNet/{win_size}_{50}_{strat}_{seed}_strides.csv', index=False)

        # print(results_df.mean(numeric_only=True).round(3)*100)
        
if __name__ == '__main__':
    main()