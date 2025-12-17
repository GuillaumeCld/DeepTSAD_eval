from eval import Evaluator
from training import Trainer
import pandas as pd
from types import SimpleNamespace

from models import Linear
from tqdm import tqdm
import tools
import numpy as np
import torch
import random
from procedure import train_and_evaluate


def main():

    # fix seed for reproducibility

    seed = 3

    path = 'Datasets/TSB-AD-U/'
    file_list = 'Datasets/File_List/TSB-AD-U-Eva-Full.csv'
    file_list = pd.read_csv(file_list)['file_name'].values

    win_size = 32

    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    results = []
    for filename in tqdm(file_list):
        model = Linear.Model(config)
        data_train, data, labels = tools.read_file(path, filename)
        trainer.train_masked(model, data_train, 20, mode="middle")
            


        reconstruction_error = evaluator.reconstruction_error(data, model, win_size)

        rank = tools.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        metrics = evaluator.metrics_fnc(
        reconstruction_error, labels, slidingWindow=rank)

        result = {'filename': filename}
        result.update(metrics)
        results.append(result)
    
        results_df = pd.DataFrame(results)

        results_df.to_csv(f'results/Linear/32_masked_points_{seed}.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)
if __name__ == '__main__':
    main()