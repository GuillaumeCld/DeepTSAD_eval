from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import DLinear
from tqdm import tqdm

import numpy as np
import torch
import random
import tools


from procedure import train_and_evaluate


def main(seed):

    # fix seed for reproducibility

    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = 'Datasets/TSB-AD-U/'
    file_list = 'Datasets/File_List/TSB-AD-U-Eva.csv'
    file_list = pd.read_csv(file_list)['file_name'].values

    win_size = 96

    moving_avg = int(win_size * 0.25)
    moving_avg = moving_avg + 1 if moving_avg % 2 == 0 else moving_avg

    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
        label_len=win_size,
        moving_avg=moving_avg,
        dropout=0.1,
        enc_in=1,
    )

    trainer = Trainer(
        batch_size=1024,
        lr=1e-2,
        device='cuda',
        win_size=win_size,
        validation_size=0.2
    )
    evaluator = Evaluator(
        batch_size=8096,
        device='cuda',
        metrics='restr',
        strategy='disjoint'
    )
    results = []
    for filename in tqdm(file_list):
        model = DLinear.Model(config)
        metrics = train_and_evaluate(
            path,
            filename,
            model,
            trainer,
            evaluator,
            win_size=win_size,
            epochs=20
        )
        # _, data, labels = tools.read_file(path, filename)

        # relative_error = evaluator.relative_reconstruction_error(
        #     data, model, win_size)

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
        # results_df.to_csv(f'results/DLinear/eval_hp_disjoint_{seed}.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)


if __name__ == '__main__':
    for seed in [4]:#, 4, 5, 6, 7]:
        main(seed)
