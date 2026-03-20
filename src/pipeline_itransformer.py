from eval import Evaluator
from training import Trainer
import models
import pandas as pd
import os
from types import SimpleNamespace

from models import iTransformer
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



def main():


   

    path = 'Datasets/TSB-AD-U/'
    file_list = 'Datasets/File_List/TSB-AD-U-Eva-Full.csv'
    file_list = pd.read_csv(file_list)['file_name'].values

    win_size = 32
    lr = 1e-4

    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
        label_len=win_size,  # unused
        pred_len=0,   # no forecasting for reconstruction
        d_model=8,
        d_ff=32,   
        factor=3,    
        e_layers=4,    # 
        enc_in=1,      # univariate input
        dec_in=1,      # univariate input
        n_heads=4,
        activation='gelu',
        embed="fixed",  
        freq='h',       
        dropout=0.1,   # dropout rate
    )
    

    
    trainer = Trainer(
        batch_size=1024,
        lr=lr,
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

    for seed in range(0, 5, 1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        results = []
        for filename in tqdm(file_list):
            model = iTransformer.Model(config)
            metrics = train_and_evaluate(
                path,
                filename,
                model,
                trainer,
                evaluator,
                win_size=win_size,
                epochs=20
            )
            result = {'filename': filename}
            result.update(metrics)
            results.append(result)
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'results/iTransformer/lr{lr}_ws{win_size}.csv', index=False)

        print(results_df.mean(numeric_only=True).round(3)*100)
if __name__ == '__main__':
    main()