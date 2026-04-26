"""Usage:
python src/tuning.py
python src/tuning.py --target-models AutoEncoder TimesNet
python src/tuning.py --dataset-path Datasets/TSB-AD-M/
python src/tuning.py --target-models DLinear --dataset-path Datasets/TSB-AD-U/
"""

import os
import random
import argparse
import time
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import torch

from eval import Evaluator
from models import AutoEncoder, Autoformer, DLinear, FEDformer, TimesNet, Transformer
from training import Trainer

# ----------------------
# Global cache
# ----------------------
DATA_CACHE = {}
TUNING_FILE_LIST = pd.read_csv(
    "Datasets/File_List/TSB-AD-U-Tuning.csv"
)["file_name"].values

COMMON_SEARCH_SPACE = {
    "win_size": [32, 64, 96],
    "lr": [1e-4, 1e-3, 1e-2],
    "strategy": ["overlapping", "disjoint"],
    "architecture_size": ["small", "medium", "large"],
    "lr_mode": ["constant", "scheduled"],
}

ARCHITECTURE_PRESETS = {
    "TimesNet": {
        "small": {
            "top_k": 3,
            "d_model": 8,
            "d_ff": 16,
            "num_kernels": 4,
            "e_layers": 1,
        },
        "medium": {
            "top_k": 5,
            "d_model": 16,
            "d_ff": 32,
            "num_kernels": 6,
            "e_layers": 2,
        },
        "large": {
            "top_k": 7,
            "d_model": 32,
            "d_ff": 64,
            "num_kernels": 8,
            "e_layers": 3,
        },
    },
    "DLinear": {
        "small": {"moving_avg_ratio": 0.1},
        "medium": {"moving_avg_ratio": 0.25},
        "large": {"moving_avg_ratio": 0.5},
    },
    "Transformer": {
        "small": {
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
            "n_heads": 2,
        },
        "medium": {
            "d_model": 16,
            "d_ff": 32,
            "e_layers": 2,
            "n_heads": 2,
        },
        "large": {
            "d_model": 32,
            "d_ff": 64,
            "e_layers": 3,
            "n_heads": 4,
        },
    },
    "FEDformer": {
        "small": {
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
            "n_heads": 2,
            "moving_avg": 15,
        },
        "medium": {
            "d_model": 16,
            "d_ff": 32,
            "e_layers": 1,
            "n_heads": 2,
            "moving_avg": 25,
        },
        "large": {
            "d_model": 32,
            "d_ff": 64,
            "e_layers": 2,
            "n_heads": 4,
            "moving_avg": 25,
        },
    },
    "Autoformer": {
        "small": {
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
            "n_heads": 2,
            "factor": 3,
            "moving_avg": 15,
        },
        "medium": {
            "d_model": 16,
            "d_ff": 32,
            "e_layers": 1,
            "n_heads": 2,
            "factor": 3,
            "moving_avg": 25,
        },
        "large": {
            "d_model": 32,
            "d_ff": 64,
            "e_layers": 2,
            "n_heads": 4,
            "factor": 5,
            "moving_avg": 25,
        },
    },
    "AutoEncoder": {
        "small": {
            "latent_ratio": 0.25,
            "hidden_ratios": [0.75],
        },
        "medium": {
            "latent_ratio": 0.4,
            "hidden_ratios": [0.9, 0.6],
        },
        "large": {
            "latent_ratio": 0.5,
            "hidden_ratios": [1.0, 0.75, 0.5],
        },
    },
}


# ----------------------
# Model config builders
# ----------------------
def build_dlinear_config(params):
    moving_avg = int(params["win_size"] * params["moving_avg_ratio"])
    if moving_avg % 2 == 0:
        moving_avg += 1

    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=params["win_size"],
        label_len=params["win_size"],
        moving_avg=moving_avg,
        dropout=0.1,
        enc_in=1,
    )


def build_timesnet_config(params):
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=params["win_size"],
        pred_len=0,
        label_len=params["win_size"],
        dropout=0.1,
        enc_in=1,
        c_out=1,
        top_k=params["top_k"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        num_kernels=params["num_kernels"],
        e_layers=params["e_layers"],
        embed="timeF",
        freq="t",
    )


def build_transformer_config(params):
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=params["win_size"],
        label_len=params["win_size"],
        pred_len=0,
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        factor=3,
        e_layers=params["e_layers"],
        d_layers=1,
        enc_in=1,
        dec_in=1,
        c_out=1,
        n_heads=params["n_heads"],
        activation="gelu",
        moving_avg=25,
        embed="fixed",
        freq="t",
        dropout=0.1,
        down_sampling_window=3,
        channel_independence=True,
        decomp_method="moving_avg",
        down_sampling_layers=2,
        use_norm=False,
        down_sampling_method="avg",
    )


def build_fedformer_config(params):
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=params["win_size"],
        label_len=params["win_size"],
        pred_len=0,
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        e_layers=params["e_layers"],
        d_layers=1,
        enc_in=1,
        dec_in=1,
        c_out=1,
        n_heads=params["n_heads"],
        activation="gelu",
        moving_avg=params["moving_avg"],
        embed="fixed",
        freq="t",
        dropout=0.1,
    )


def build_autoformer_config(params):
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=params["win_size"],
        label_len=params["win_size"],
        pred_len=0,
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        factor=params["factor"],
        e_layers=params["e_layers"],
        d_layers=1,
        enc_in=1,
        dec_in=1,
        c_out=1,
        n_heads=params["n_heads"],
        activation="gelu",
        moving_avg=params["moving_avg"],
        embed="fixed",
        freq="t",
        dropout=0.1,
    )


def build_autoencoder_config(params):
    latent_len = max(2, int(params["win_size"] * params["latent_ratio"]))

    hidden_dims = []
    if "hidden_ratios" in params:
        hidden_dims = [
            min(max(2, int(round(params["win_size"] * float(ratio)))), max(2, params["win_size"] - 1))
            for ratio in params.get("hidden_ratios", [])
        ]
    elif "hidden_dims" in params:
        hidden_dims = [
            min(max(2, int(width)), max(2, params["win_size"] - 1))
            for width in params.get("hidden_dims", [])
        ]

    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=params["win_size"],
        enc_in=1,
        latent_len=latent_len,
        hidden_dims=hidden_dims,
        hidden_ratios=params.get("hidden_ratios", []),
        activation="relu",
    )


MODEL_SPECS = {
    "TimesNet": {
        "model_class": TimesNet.Model,
        "search_space": {},
        "build_config": build_timesnet_config,
    },
    "DLinear": {
        "model_class": DLinear.Model,
        "search_space": {},
        "build_config": build_dlinear_config,
    },
    "Transformer": {
        "model_class": Transformer.Model,
        "search_space": {},
        "build_config": build_transformer_config,
    },
    "FEDformer": {
        "model_class": FEDformer.Model,
        "search_space": {},
        "build_config": build_fedformer_config,
    },
    "Autoformer": {
        "model_class": Autoformer.Model,
        "search_space": {},
        "build_config": build_autoformer_config,
    },
    "AutoEncoder": {
        "model_class": AutoEncoder.Model,
        "search_space": {},
        "build_config": build_autoencoder_config,
    },
}

TUNING_SETTINGS = {
    "target_models": ["DLinear", "AutoEncoder", "TimesNet", "FEDformer", "Autoformer"],
    "dataset_path": "Datasets/TSB-AD-U/",
    "seeds": [0, 1, 2],
    "epoch_candidates": [10, 20, 30, 50],
    "n_jobs": 1,
    "scheduled_lr_scheduler": "plateau",
    "scheduled_lr_scheduler_kwargs": {
        "patience": 5,
        "factor": 0.5,
        "min_lr": 1e-5,
    },
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
        labels = df["Label"].astype(int).to_numpy()

        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std == 0, 1e-8, std)
        data = (data - mean) / std

        train_index = int(filename.split(".")[0].split("_")[-3])

        data_train = data[:train_index]
        data_test = data

        DATA_CACHE[key] = (data_train, data_test, labels)

    return DATA_CACHE[key]


def merged_search_space(model_name):
    return {
        **COMMON_SEARCH_SPACE,
        **MODEL_SPECS[model_name]["search_space"],
    }


def suggest_params(trial, search_space):
    params = {}
    for key, values in search_space.items():
        params[key] = trial.suggest_categorical(key, values)
    return params


def apply_architecture_preset(model_name, params):
    size = params["architecture_size"]
    presets = ARCHITECTURE_PRESETS.get(model_name, {})
    if size not in presets:
        raise ValueError(
            f"Missing architecture preset for model '{model_name}' and size '{size}'"
        )
    return {
        **params,
        **presets[size],
    }


def scheduled_lr_kwargs(max_epochs, scheduler_name, base_kwargs=None):
    del max_epochs
    del scheduler_name
    return dict(base_kwargs or {})


def make_objective(model_name):
    spec = MODEL_SPECS[model_name]
    model_class = spec["model_class"]
    build_config = spec["build_config"]
    search_space = merged_search_space(model_name)

    def objective(trial):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trial_start_time = time.perf_counter()
        params = suggest_params(trial, search_space)
        params = apply_architecture_preset(model_name, params)
        config = build_config(params)
        epoch_candidates = TUNING_SETTINGS["epoch_candidates"]

        use_scheduler = params["lr_mode"] == "scheduled"
        scheduler_name = (
            TUNING_SETTINGS["scheduled_lr_scheduler"] if use_scheduler else "none"
        )
        scheduler_kwargs = (
            scheduled_lr_kwargs(
                max(epoch_candidates),
                scheduler_name,
                TUNING_SETTINGS["scheduled_lr_scheduler_kwargs"],
            ) if use_scheduler else {}
        )

        epoch_seed_scores = {epoch: [] for epoch in epoch_candidates}
        for seed in TUNING_SETTINGS["seeds"]:
            set_seed(seed)

            trainer = Trainer(
                batch_size=1024,
                lr=params["lr"],
                device=device,
                win_size=params["win_size"],
                validation_size=0.2,
                lr_scheduler=scheduler_name,
                lr_scheduler_kwargs=scheduler_kwargs,
            )

            evaluator = Evaluator(
                batch_size=10000,
                device=device,
                metrics="restr",
                strategy=params["strategy"],
            )

            seed_epoch_scores = {epoch: [] for epoch in epoch_candidates}
            for filename in TUNING_FILE_LIST:
                data_train, data_test, labels = load_dataset(
                    TUNING_SETTINGS["dataset_path"],
                    filename,
                )
                model = model_class(config).to(device)
                previous_epoch = 0
                for epochs in epoch_candidates:
                    epochs_to_train = epochs - previous_epoch
                    trainer.train(model, data_train, epochs_to_train)
                    previous_epoch = epochs

                    metrics = evaluator.evaluate(
                        data_test,
                        labels,
                        model,
                        params["win_size"],
                        stride=1,
                    )

                    score = metrics.get("AUC-PR", None) or list(metrics.values())[0]
                    seed_epoch_scores[epochs].append(score)

            for epoch in epoch_candidates:
                epoch_seed_scores[epoch].append(float(np.mean(seed_epoch_scores[epoch])))

        epoch_mean_scores = {
            epoch: float(np.mean(scores))
            for epoch, scores in epoch_seed_scores.items()
        }
        best_epoch = max(epoch_mean_scores, key=epoch_mean_scores.get)
        seed_scores = epoch_seed_scores[best_epoch]

        final_score = np.round(float(np.mean(seed_scores)), 2)
        trial.set_user_attr("model", model_name)
        trial.set_user_attr("architecture_size", params["architecture_size"])
        trial.set_user_attr("lr_mode", params["lr_mode"])
        trial.set_user_attr("lr_scheduler", scheduler_name)
        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("trained_epochs", int(best_epoch))
        trial.set_user_attr(
            "computation_time_seconds",
            float(time.perf_counter() - trial_start_time),
        )
        return float(final_score)

    return objective


def run_model_study(model_name):
    search_space = merged_search_space(model_name)
    study_name = f"HP_{model_name}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.GridSampler(search_space),
    )
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.FAIL: 
            study.enqueue_trial(trial.params)

    study.optimize(
        make_objective(model_name),
        n_jobs=TUNING_SETTINGS["n_jobs"],
    )

    print(f"[{model_name}] Best params:", study.best_params)
    print(f"[{model_name}] Best score:", study.best_value)

    os.makedirs("results", exist_ok=True)
    study.trials_dataframe().to_csv(f"results/{study_name}.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for selected models.")
    parser.add_argument(
        "--target-models",
        nargs="+",
        default=TUNING_SETTINGS["target_models"],
        help="Models to tune. Example: --target-models AutoEncoder TimesNet",
    )
    parser.add_argument(
        "--dataset-path",
        default=TUNING_SETTINGS["dataset_path"],
        help="Path to dataset directory used during tuning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_models = args.target_models
    TUNING_SETTINGS["dataset_path"] = args.dataset_path

    unknown_models = [
        model_name
        for model_name in target_models
        if model_name not in MODEL_SPECS
    ]
    if unknown_models:
        raise ValueError(
            f"Unknown models in target_models: {unknown_models}. "
            f"Available models: {list(MODEL_SPECS.keys())}"
        )

    for model_name in target_models:
        run_model_study(model_name)


if __name__ == "__main__":
    main()
