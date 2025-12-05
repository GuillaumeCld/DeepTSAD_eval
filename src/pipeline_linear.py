from eval import Evaluator
from training import Trainer
import pandas as pd
from types import SimpleNamespace

from models import Linear
from tqdm import tqdm

import numpy as np
import torch
import random
from procedure import train_and_evaluate


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
        model = Linear.Model(config)
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
        results_df.to_csv(f'results/Linear/median_{seed}.csv', index=False)

    print(results_df.mean(numeric_only=True).round(3)*100)
if __name__ == '__main__':
    main()