from eval import Evaluator
from training import Trainer
import pandas as pd
import os
from types import SimpleNamespace

from models import DLinear
from tqdm import tqdm

import numpy as np
import torch
import random

from procedure import train_and_evaluate


def main():

    # ----------------------
    # Fix seed
    # ----------------------
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------
    # Device
    # ----------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------
    # Paths
    # ----------------------
    path = 'Datasets/TSB-AD-U/'
    file_list_path = 'Datasets/File_List/TSB-AD-U-Tuning.csv'
    file_list = pd.read_csv(file_list_path)['file_name'].values

    os.makedirs("results/DLinear", exist_ok=True)

    # ----------------------
    # Hyperparameters
    # ----------------------
    win_sizes = [16, 32, 64, 96]
    lrs = [1e-2, 1e-3, 1e-4]
    epochs_list = [10, 20, 30, 50]

    all_results = []

    for win_size in win_sizes:
        for lr in lrs:

            moving_avgs = list(set([
                win_size * 3 // 4 + 1,
                win_size // 2 + 1,
                win_size // 4 + 1,
                win_size // 8 + 1
            ]))

            for moving_avg in moving_avgs:

                config = SimpleNamespace(
                    task_name='anomaly_detection',
                    seq_len=win_size,
                    label_len=win_size,
                    moving_avg=moving_avg,
                    dropout=0.1,
                    enc_in=1,
                )

                trainer = Trainer(
                    batch_size=256,
                    lr=lr,
                    device=device,
                    win_size=win_size,
                    validation_size=0.2
                )

                evaluator = Evaluator(
                    batch_size=1024,
                    device=device,
                    metrics='restr',
                    strategy='overlapping'
                )

                results = []

                for filename in tqdm(file_list):

                    model = DLinear.Model(config).to(device)

                    for i, total_epoch in enumerate(epochs_list):
                        epoch = total_epoch - (epochs_list[i-1] if i > 0 else 0)

                        metrics = train_and_evaluate(
                            path,
                            filename,
                            model,
                            trainer,
                            evaluator,
                            win_size=win_size,
                            epochs=epoch
                        )

                        result = {
                            'filename': filename,
                            'win_size': win_size,
                            'lr': lr,
                            'epochs': total_epoch,
                            'moving_avg': moving_avg
                        }

                        result.update(metrics)

                        results.append(result)
                        all_results.append(result)

                # Save per config
                pd.DataFrame(results).to_csv(
                    f'results/DLinear/hp_tuning_{evaluator.strategy}_256.csv',
                    index=False
                )

                print(f'Win size: {win_size}, LR: {lr}, Moving Avg: {moving_avg}')
                print(pd.DataFrame(results).mean(numeric_only=True).round(3) * 100)


if __name__ == '__main__':
    main()
