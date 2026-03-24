import optuna
import pandas as pd
import numpy as np
import torch
import random
import os

from types import SimpleNamespace

from models import DLinear, TimesNet  # add other models here
from training import Trainer
from eval import Evaluator
from procedure import train_and_evaluate
from tqdm import tqdm

# ----------------------
# Global cache
# ----------------------
DATA_CACHE = {}

MODEL_REGISTRY = {
    # "DLinear": DLinear.Model,
    "TimesNet": TimesNet.Model,
}


# ----------------------
# Reproducibility
# ----------------------
def set_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------
# Cached loader
# ----------------------
def load_dataset(path, filename):
    key = (path, filename)

    if key not in DATA_CACHE:
        file_path = os.path.join(path, filename)

        df = pd.read_csv(file_path).dropna()

        data = df.iloc[:, 0:-1].values.astype(float)
        labels = df['Label'].astype(int).to_numpy()

        # normalize
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std == 0, 1e-8, std)
        data = (data - mean) / std

        train_index = int(filename.split('.')[0].split('_')[-3])

        data_train = data[:train_index]
        data_test = data

        DATA_CACHE[key] = (data_train, data_test, labels)

    return DATA_CACHE[key]


def check_repeated_trial(trial):
  optuna_study = trial.study
  

  for past_trial in optuna_study.get_trials():
    if past_trial.number == trial.number:
      continue

    past_params = past_trial.params
    repeated_trial = True
    
    for key in trial.params:
      if key in past_params and trial.params[key] != past_params[key]:
        repeated_trial = False
        break 
    
    return repeated_trial
# ----------------------
# Objective
# ----------------------
def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # seeds for robustness
    seeds = [0, 1, 2]

    # ----------------------
    # Model choice
    # ----------------------
    model_name = trial.suggest_categorical(
        "model",
        list(MODEL_REGISTRY.keys())
    )

    ModelClass = MODEL_REGISTRY[model_name]

    # ----------------------
    # Shared hyperparameters
    # ----------------------
    win_size = trial.suggest_categorical("win_size", [32, 64, 96])
    lr = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2])
    epochs = trial.suggest_categorical("epochs", [10, 20, 30])

    # ----------------------
    # Model-specific params
    # ----------------------
    if model_name == "DLinear":
        moving_avg_ratio = trial.suggest_categorical(
            "moving_avg_ratio",
            [0.1, 0.25, 0.5, 0.75]
        )

        moving_avg = int(win_size * moving_avg_ratio)
        moving_avg = moving_avg + 1 if moving_avg % 2 == 0 else moving_avg

        config = SimpleNamespace(
            task_name='anomaly_detection',
            seq_len=win_size,
            label_len=win_size,
            moving_avg=moving_avg,
            dropout=0.1,
            enc_in=1,
        )
    elif model_name == "TimesNet":
        top_k = trial.suggest_categorical("top_k", [3, 5, 7])
        d_model = trial.suggest_categorical("d_model", [8, 16, 32])
        d_ff = trial.suggest_categorical("d_ff", [16, 32, 64])
        num_kernels = trial.suggest_categorical("num_kernels", [4, 6, 8])
        e_layers = trial.suggest_categorical("e_layers", [1, 2, 3])
        config = SimpleNamespace(
            task_name='anomaly_detection',
            seq_len=win_size,
            pred_len=0,
            label_len=win_size,
            dropout=0.1,
            enc_in=1,
            c_out=1,
            top_k=top_k,
            d_model=d_model,
            d_ff=d_ff,
            num_kernels=num_kernels,
            e_layers=e_layers,
            embed="timeF",
            freq="t",
        )
    else:
        raise NotImplementedError(model_name)

    if check_repeated_trial(trial):
        print(f"Trial {trial.number} is a repeated trial. Skipping...")
        raise optuna.exceptions.TrialPruned()

    # dataset
    path = 'Datasets/TSB-AD-U/'
    file_list = pd.read_csv(
        'Datasets/File_List/TSB-AD-U-Tuning.csv'
    )['file_name'].values

    seed_scores = []

    # ======================
    # LOOP OVER SEEDS
    # ======================
    for seed in seeds:
        set_seed(seed)

        trainer = Trainer(
            batch_size=1024,
            lr=lr,
            device=device,
            win_size=win_size,
            validation_size=0.2
        )

        evaluator = Evaluator(
            batch_size=2048,
            device=device,
            metrics='restr',
            strategy='overlapping'
        )

        scores = []

        for filename in tqdm(file_list):

            data = load_dataset(path, filename)

            model = ModelClass(config).to(device)

            metrics = train_and_evaluate(
                path,
                filename,
                model,
                trainer,
                evaluator,
                win_size=win_size,
                epochs=epochs,
                data=data
            )

            score = metrics.get("AUC-PR", None) or list(metrics.values())[0]
            score = np.round(score, 3)
            scores.append(score)

        # average over datasets for this seed
        seed_scores.append(np.mean(scores))

    # ======================
    # FINAL AVERAGE OVER SEEDS
    # ======================
    final_score = float(np.mean(seed_scores))

    return np.round(final_score, 3)
def main():

    os.makedirs("results", exist_ok=True)

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="maximize",
        study_name="TimesNet_big_tuning",
        storage="sqlite:///optuna.db",   # persistent + parallel-safe
        load_if_exists=True,
        pruner=pruner,
        sampler=optuna.samplers.GridSampler(
            {
                "win_size": [32, 64, 96],
                "lr": [1e-4, 1e-3, 1e-2],
                "epochs": [10, 20, 30],
                "top_k": [3, 5, 7],
                "d_model": [8, 16, 32],
                "d_ff": [16, 32, 64],
                "num_kernels": [4, 6, 8],
                "e_layers": [1, 2, 3],
                # "moving_avg_ratio": [0.1, 0.25, 0.5, 0.75],
            }
        )
    )


    study.optimize(
        objective,
        n_jobs=1   # parallel
    )

    print("Best model:", study.best_params["model"])
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)

    study.trials_dataframe().to_csv("results/optuna_results.csv", index=False)


if __name__ == "__main__":
    main()