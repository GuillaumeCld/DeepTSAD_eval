"""
Hyperparameter tuning with Optuna optimizing AUC-PR.

Usage:
1)  python src/optuning.py --config configs/experiment.conf --model_config configs/TimesNet.conf

2)  optuna-dashboard sqlite:///results/optuna/aucpr.db
Notes:
- Expects flat .conf files (no sections). Example:
    seed = 42
    device = cuda
    ...
- file_list and metrics are stored as comma-separated strings in the config.
- The AUC-PR metric key returned by `train_and_evaluate` is assumed to be one of:
    auc_pr, auprc, average_precision, auc-pr
  (auto-detected in that order). Adjust `AUC_PR_CANDIDATES` if needed.
"""

from eval import Evaluator
from training import Trainer
import pandas as pd
import models  # package: models/<ModelName>.py with class Model inside
from tqdm import tqdm

import numpy as np
import torch
import random
from procedure import train_and_evaluate

import configparser
from types import SimpleNamespace
import argparse
import os
import copy
import importlib

import optuna


# ----------------------------
# Config loading (flat .conf)
# ----------------------------
def load_config(path: str) -> SimpleNamespace:
    """
    Loads a flat key=value config file into a SimpleNamespace with auto-casting.
    Supports files with no [SECTION] header by injecting [DEFAULT].
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.lstrip().startswith("["):
        content = "[DEFAULT]\n" + content

    parser = configparser.ConfigParser()
    parser.read_string(content)

    def cast(value: str):
        v = value.strip()
        # bool
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        # int / float
        for fn in (int, float):
            try:
                return fn(v)
            except ValueError:
                pass
        return v

    cfg = {k: cast(v) for k, v in parser["DEFAULT"].items()}
    return SimpleNamespace(**cfg)


def split_csv_string(s) -> list[str]:
    """Turns 'a,b, c' into ['a','b','c'] (safe for None/empty)."""
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Model loading
# ----------------------------
def build_model(model_config):
    """
    Loads models/<model_name>.py and returns Model(model_config).
    model_config.model_name must match the module filename (case-sensitive on Linux).
    Example: model_name = TimesNet -> models/TimesNet.py
    """
    module = importlib.import_module(f"models.{model_config.model_name}")
    return module.Model(model_config)


# ----------------------------
# Experiment runner
# ----------------------------
AUC_PR_CANDIDATES = ("AUC-PR", "auc_pr", "auprc", "average_precision", "auc-pr")


def run_experiment(config, model_config, save_csv_path: str | None = None, show_progress: bool = False):
    """
    Train + evaluate over all files in config.file_list.
    Returns:
      mean_auc_pr (float), results_df (DataFrame)
    """
    model_config.seq_len = config.win_size
    model_config.label_len = config.win_size
    trainer = Trainer(
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        win_size=config.win_size,
        validation_size=getattr(config, "validation_size", 0.2),
    )

    evaluator = Evaluator(
        batch_size=config.batch_size,
        device=config.device,
        metrics="restr",  # list[str]
        strategy=config.strategy,
    )

    results = []
    file_list = pd.read_csv(config.file_list)['file_name'].values


    for filename in file_list:

        model = build_model(model_config)

        metrics = train_and_evaluate(
            config.path,
            filename,
            model,
            trainer,
            evaluator,
            win_size=config.win_size,
            epochs=config.epochs,
        )

        row = {"filename": filename}
        row.update(metrics)
        results.append(row)

    results_df = pd.DataFrame(results)

    if save_csv_path is not None:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        results_df.to_csv(save_csv_path, index=False)



    mean_auc_pr = float(results_df["AUC-PR"].mean())
    return mean_auc_pr, results_df


# ----------------------------
# Optuna objective
# ----------------------------
def make_objective(base_config, base_model_config, trials_dir="results/optuna"):
    """
    Creates an Optuna objective function that maximizes mean AUC-PR.
    Edit the suggest_* calls to match your model/training knobs.
    """
    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)
        model_config = copy.deepcopy(base_model_config)

        # ---- Sample training hyperparameters ----
        config.lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        config.win_size = trial.suggest_categorical("win_size", [16, 32, 64, 96, 128])

        # Optional: budgeted epochs for tuning (uncomment if you want faster tuning)
        # config.epochs = trial.suggest_categorical("epochs", [10, 20, 50])

        # ---- Sample model hyperparameters (only if present) ----
        # These names must match what your Model reads from model_config.
        # if hasattr(model_config, "hidden_dim"):
        #     model_config.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        # if hasattr(model_config, "dropout"):
        #     model_config.dropout = trial.suggest_float("dropout", 0.0, 0.5)
        # if hasattr(model_config, "num_layers"):
        #     model_config.num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4])

        # Re-seed per trial
        base_seed = int(getattr(config, "seed", 42))
        seed_everything(base_seed + trial.number)

        # Save per-trial metrics
        save_path = os.path.join(trials_dir, str(model_config.model_name), f"trial_{trial.number}.csv")

        mean_auc_pr, _df = run_experiment(
            config,
            model_config,
            save_csv_path=save_path,
            show_progress=False,
        )

        return mean_auc_pr  # maximize

    return objective


def tune_with_optuna(config, model_config, n_trials=10, study_name="test"):
    sampler = optuna.samplers.TPESampler(seed=int(getattr(config, "seed", 42)))
    storage = "sqlite:///results/optuna/aucpr.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    def log_callback(study, trial):
        print(
            f"[trial {trial.number}] value={trial.value:.6f}  best={study.best_value:.6f}  params={trial.params}"
        )

    objective = make_objective(config, model_config)
    study.optimize(objective, n_trials=n_trials, callbacks=[log_callback], show_progress_bar=True)

    print("\n==== Optuna Results ====")
    print("Best mean AUC-PR:", study.best_value)
    print("Best params:", study.best_params)
    return study


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to flat training/eval .conf file")
    ap.add_argument("--model_config", required=True, help="Path to flat model .conf file")
    ap.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    ap.add_argument("--study_name", type=str, default="aucpr_tuning", help="Optuna study name")
    ap.add_argument("--subset_files", type=int, default=0, help="If >0, tune on only first N files")
    args = ap.parse_args()

    config = load_config(args.config)
    model_config = load_config(args.model_config)

    # file_list and metrics are strings -> convert once
    # config.file_list = split_csv_string(getattr(config, "file_list", ""))
    # config.metrics = split_csv_string(getattr(config, "metrics", ""))

    if args.subset_files and args.subset_files > 0:
        config.file_list = config.file_list[: args.subset_files]

    # Ensure base output dir exists
    os.makedirs("results/optuna", exist_ok=True)

    # Optional: quick sanity check — uncomment to see which AUC-PR key is produced
    # seed_everything(int(getattr(config, "seed", 42)))
    # mean_auc_pr, df = run_experiment(config, model_config, show_progress=True)
    # print("Columns:", list(df.columns))
    # print("Mean AUC-PR:", mean_auc_pr)
    # return

    tune_with_optuna(config, model_config, n_trials=args.trials, study_name=args.study_name)


if __name__ == "__main__":
    main()
