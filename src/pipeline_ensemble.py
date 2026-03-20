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
from vus_torch import VUSTorch


def main():

    # fix seed for reproducibility

    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = 'Datasets/TSB-AD-U/'
    file_list = 'Datasets/File_List/TSB-AD-U-Eva.csv'
    file_list = pd.read_csv(file_list)['file_name'].values

    win_size = 96

    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
    )

    # dlinear
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

    # timesnet
    # config = SimpleNamespace(
    #     task_name='anomaly_detection',
    #     seq_len=win_size,
    #     label_len=win_size,  # unused
    #     pred_len=0,   # no forecasting for reconstruction
    #     top_k=3,
    #     d_model=8,
    #     d_ff=16,
    #     num_kernels=6,  # number of kernels in InceptionBlock
    #     e_layers=1,    # number of TimesNet blocks
    #     embed='timeF',
    #     freq='t',
    #     dropout=0.1,   # dropout rate
    #     enc_in=1,      # univariate input
    #     c_out=1,       # univariate output
    # )

    # vus = VUSTorch(slope_size=128, device='cuda')

    trainer = Trainer(
        batch_size=1024,
        lr=1e-2,
        device='cuda',
        win_size=win_size,
        validation_size=0.2
    )

    strat = "overlapping"  # "overlapping" or "disjoint"

    results = []

    evaluator = Evaluator(
        batch_size=1024,
        device="cuda",
        metrics="restr",
        strategy=strat,
    )
    n_seeds = 5
    for filename in tqdm(file_list):
        # if int(filename[:3]) < 691:
        #     continue  # skip training files
        model_errors = []  # reconstruction_error per seed
        # Load once per file
        data_train, data, labels = tools.read_file(path, filename)
        trainer.win_size = win_size
        model_relative_errors = torch.zeros(data.shape[0], device='cuda')

        for seed in range(3, 3+n_seeds, 1):
            # Seeding
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Train a model for this seed
            model = DLinear.Model(config)
            trainer.train(model, data_train, 30)

            # Reconstruction error from this model
            reconstruction_error = evaluator.reconstruction_error(
                data, model, win_size)
            model_errors.append(np.asarray(reconstruction_error))

            relative_error = evaluator.relative_reconstruction_error(
                data, model, win_size)
            model_relative_errors += relative_error

            del model
            torch.cuda.empty_cache()

        # ---- Ensemble reconstruction: MEAN over seeds ----
        reconstruction_error_ens = np.mean(
            np.stack(model_errors, axis=0), axis=0)

        # relative reconstruction errors
        relative_error /= n_seeds
        # on normal data points
        # / (data[labels == 0, 0] + 1e-8)
        # relative_normal = reconstruction_error_ens[labels == 0]
        # on abnormal data points
        # / (data[labels == 1, 0] + 1e-8)
        # relative_abnormal = reconstruction_error_ens[labels == 1]

        # Metrics computed ONCE using ensemble error
        rank = tools.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        metrics = evaluator.metrics_fnc(
            reconstruction_error_ens, labels, slidingWindow=rank
        )

        # rel_erruer_normal = relative_error[labels == 0].mean()
        # rel_erreur_anomalie = relative_error[labels == 1].mean()
        # top_1_normal = relative_error[labels == 0].max()
        # top_1_anomalie = relative_error[labels == 1].max()


        result = {'filename': filename,
                #   'rel_normal': rel_erruer_normal.item(),
                #   'rel_abnormal': rel_erreur_anomalie.item(),
                #   'top1_normal': top_1_normal.item(),
                #   'top1_abnormal': top_1_anomalie.item()
                  }
        result.update(metrics)
        results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            f"results/DLinear/eval_ensemble_hp.csv",
            index=False
        )
    print(results_df.mean(numeric_only=True))


if __name__ == '__main__':
    main()
