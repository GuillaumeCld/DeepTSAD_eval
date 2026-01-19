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

    win_size = 0

    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
    )
    

    # dlinear
    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=0,
        label_len=win_size,  # unused
        pred_len=win_size,   # no forecasting for reconstruction
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




    
    strat = "overlapping"  # "overlapping" or "disjoint"


    results = []

    evaluator = Evaluator(
        batch_size=1024,
        device="cuda",
        metrics="restr",
        strategy=strat,
    )
    win_sizes = [16, 32, 64, 96, 128]
    for filename in tqdm(file_list):
        model_errors = []  # reconstruction_error per seed

        # Load once per file
        data_train, data, labels = tools.read_file(path, filename)

        for seed in range(0, 5):
            # Seeding
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            win_size = win_sizes[seed]
            config.seq_len = win_size

            # Train a model for this seed
            model = DLinear.Model(config)
            trainer = Trainer(
                batch_size=1024,
                lr=1e-2,
                device='cuda',
                win_size=win_size,
                validation_size=0.2
            )
                    
            trainer.train(model, data_train, 20)


            # Reconstruction error from this model
            reconstruction_error = evaluator.reconstruction_error(data, model, win_size)
            model_errors.append(np.asarray(reconstruction_error))

            del model
            torch.cuda.empty_cache()

        # ---- Ensemble reconstruction: MEAN over seeds ----
        reconstruction_error_ens = np.mean(np.stack(model_errors, axis=0), axis=0)

        # Metrics computed ONCE using ensemble error
        rank = tools.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        metrics = evaluator.metrics_fnc(
            reconstruction_error_ens, labels, slidingWindow=rank
        )

        result = {"filename": filename}
        result.update(metrics)
        results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            f"results/DLinear/ensemble_ML_{win_size}_{20}_{strat}.csv",
            index=False
        )

if __name__ == '__main__':
    main()