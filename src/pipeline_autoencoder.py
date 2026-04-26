from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import AutoEncoder
from tqdm import tqdm

import numpy as np
import torch
import random



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
                       win_size=None,
                       epochs=20):
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

    return evaluator.evaluate(data, labels, model, win_size)



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

    latent_len = max(2, int(96 * 0.4))

    hidden_dims = [
        min(max(2, int(width)), max(2, win_size -1))
        for width in [48, 24]
    ]
    config = SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=win_size,
        enc_in=1,
        latent_len=latent_len,
        hidden_dims=hidden_dims,
        activation="relu",
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
        model = AutoEncoder.Model(config)
        metrics = train_and_evaluate(
            path,
            filename,
            model,
            trainer,
            evaluator,
            win_size=win_size,
            epochs=50
        )
        result = {'filename': filename}
        result.update(metrics)
        results.append(result)
        
        results_df = pd.DataFrame(results)
        # results_df.to_csv(f'results/AutoEncoder/eval_hp_{seed}.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)
if __name__ == '__main__':
    for seed in range(3, 4, 1):
        main(seed)