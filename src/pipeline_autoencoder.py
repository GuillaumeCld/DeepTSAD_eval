from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace
import src.tools as utils


from models import AutoEncoder
from tqdm import tqdm

import numpy as np
import torch
import random
import time


def _read_file(path, filename):
    file_path = os.path.join(path, filename)

    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    # normalize data globally
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std

    train_index = filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    return data_train, data, label


def train_and_evaluate(path,
                       filename,
                       model,
                       trainer,
                       evaluator,
                       win_size,
                       epochs,
                       stride):
    """
    Read dataset from filename, train model and evaluate.
    trainer and evaluator should be instantiated by the caller.
    """
    data_train, data, labels = _read_file(path, filename)

    if win_size is None:
        win_size = trainer.win_size
    else:
        trainer.win_size = win_size

    trainer.train(model, data_train, epochs)

    return evaluator.evaluate(data, labels, model, win_size, stride)


def main(seed):

    # fix seed for reproducibility

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    path = 'Datasets/TSB-AD-U/'
    file_list = 'Datasets/File_List/TSB-AD-U-Eva.csv'
    file_list = pd.read_csv(file_list)['file_name'].values

    win_size = 96
    epochs = 50
    hidden_ratios = [0.5, 0.25]
    config = SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=win_size,
        enc_in=1,
        hidden_ratios=hidden_ratios,
        activation="relu",
    )

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
    evaluator = Evaluator(
        batch_size=1000,
        device='cuda',
        metrics='restr',
        strategy='overlapping'
    )
    strides = [1, 12, 24, 36, 48, 60, 72, 84, 96]
    results = []
    for filename in tqdm(file_list):
        model = AutoEncoder.Model(config)

        data_train, data, labels = _read_file(path, filename)
        rank = utils.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

        trainer.train(model, data_train, epochs)
        for stride in strides:
            start_time = time.time()
            reconstruction_error = evaluator.reconstruction_error(
                data, model, win_size, stride)
            end_time = time.time()

            metrics = evaluator.metrics_fnc(
                reconstruction_error, labels, slidingWindow=rank)

            result = {'filename': filename, 'stride': stride,
                      'execution_time': end_time - start_time}
            result.update(metrics)
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f'results/AutoEncoder/paper_run_stride_{seed}.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)


if __name__ == '__main__':
    for seed in range(3, 8, 1):
        main(seed)
