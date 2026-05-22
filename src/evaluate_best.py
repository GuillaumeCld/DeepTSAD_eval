"""Usage:
python src/evaluate_best.py
python src/evaluate_best.py --target-models DLinear TimesNet
python src/evaluate_best.py --dataset-path Datasets/TSB-AD-U/ --file-list Datasets/File_List/TSB-AD-U-Eva.csv
python src/evaluate_best.py --tuning-results-dir results --output-dir results/evaluation_best
"""

import argparse
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from eval import Evaluator
from models import AutoEncoder, Autoformer, DLinear, FEDformer, TimesNet, Transformer
from procedure import train_and_evaluate
from training import Trainer
import tools as utils

DATA_CACHE = {}

DEFAULT_TARGET_MODELS = ["DLinear", "AutoEncoder", "TimesNet", "FEDformer", "Autoformer"]
DEFAULT_SEEDS = [3,4,5,6,7]
DEFAULT_TUNING_RESULTS_DIR = "results"
DEFAULT_OUTPUT_DIR = "results/evaluation_best"
DEFAULT_DATASET_PATH = "Datasets/TSB-AD-U/"
DEFAULT_FILE_LIST = "Datasets/File_List/TSB-AD-U-Eva-Full.csv"
DEFAULT_TUNING_MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EVAL_BATCH_SIZE = 10000
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_SCHEDULED_LR_SCHEDULER = "plateau"
DEFAULT_SCHEDULED_LR_SCHEDULER_KWARGS = {
    "patience": 5,
    "factor": 0.5,
    "min_lr": 1e-5,
}

ARCHITECTURE_PRESETS = {
    "TimesNet": {
        "small": {
            "top_k": 3,
            "d_model": 8,
            "d_ff": 16,
            "num_kernels": 3,
            "e_layers": 1,
        },
        "medium": {
            "top_k": 3,
            "d_model": 12,
            "d_ff": 24,
            "num_kernels": 4,
            "e_layers": 1,
        },
        "large": {
            "top_k": 3,
            "d_model": 16,
            "d_ff": 32,
            "num_kernels": 6,
            "e_layers": 2,
        },
    },
    "DLinear": {
        "small": {"moving_avg_ratio": 0.1},
        "medium": {"moving_avg_ratio": 0.25},
        "large": {"moving_avg_ratio": 0.5},
    },
    "Transformer": {
        "small": {"d_model": 8, "d_ff": 16, "e_layers": 1, "n_heads": 2},
        "medium": {"d_model": 16, "d_ff": 32, "e_layers": 2, "n_heads": 2},
        "large": {"d_model": 32, "d_ff": 64, "e_layers": 3, "n_heads": 4},
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
        "small": {"hidden_ratios": [0.5, 0.25]},
        "medium": {"hidden_ratios": [0.75, 0.5, 0.25]},
        "large": {"hidden_ratios": [1, 0.75, 0.5, 0.25]},
    },
}

MODEL_SPECS = {
    "TimesNet": {"model_class": TimesNet.Model, "build_config": None},
    "DLinear": {"model_class": DLinear.Model, "build_config": None},
    "Transformer": {"model_class": Transformer.Model, "build_config": None},
    "FEDformer": {"model_class": FEDformer.Model, "build_config": None},
    "Autoformer": {"model_class": Autoformer.Model, "build_config": None},
    "AutoEncoder": {"model_class": AutoEncoder.Model, "build_config": None},
}


def set_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def scheduled_lr_kwargs(max_epochs, scheduler_name, base_kwargs=None):
    del max_epochs
    del scheduler_name
    return dict(base_kwargs or {})


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
        top_k=3,
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

    print(f"Building AutoEncoder config with params: {params}")
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=params["win_size"],
        enc_in=1,
        hidden_ratios=params["hidden_ratios"],
        activation="relu",
    )


BUILD_CONFIGS = {
    "TimesNet": build_timesnet_config,
    "DLinear": build_dlinear_config,
    "Transformer": build_transformer_config,
    "FEDformer": build_fedformer_config,
    "Autoformer": build_autoformer_config,
    "AutoEncoder": build_autoencoder_config,
}


def resolve_trials_csv(tuning_results_dir, model_name):
    candidates = [
        os.path.join(tuning_results_dir, f"FHP_{model_name}.csv"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find tuning results for '{model_name}'. Tried: {candidates}"
    )


def load_best_params(tuning_results_dir, model_name, strategy=None):
    trials_csv = resolve_trials_csv(tuning_results_dir, model_name)
    trials_df = pd.read_csv(trials_csv)

    if "state" in trials_df.columns:
        completed = trials_df[trials_df["state"].astype(
            str).str.lower() == "complete"]
        if not completed.empty:
            trials_df = completed

    if strategy is not None:
        strategy_col = None
        if "params_strategy" in trials_df.columns:
            strategy_col = "params_strategy"
        elif "user_attrs_strategy" in trials_df.columns:
            strategy_col = "user_attrs_strategy"

        if strategy_col is not None:
            filtered_df = trials_df[
                trials_df[strategy_col].astype(str).str.lower() == str(strategy).lower()
            ]
            if not filtered_df.empty:
                trials_df = filtered_df
            else:
                available_strategies = sorted(
                    {
                        str(value)
                        for value in trials_df[strategy_col].dropna().astype(str).unique()
                    }
                )
                raise ValueError(
                    f"No completed trials found for model '{model_name}' and strategy '{strategy}'. "
                    f"Available strategies in '{trials_csv}': {available_strategies}"
                )

    if "value" in trials_df.columns:
        max_value = trials_df["value"].astype(float).max()
        best_candidates = trials_df[trials_df["value"].astype(float) == max_value]
        
        # On ties, pick the one with lowest execution time
        exec_time_col = None
        if "user_attrs_computation_time_seconds" in best_candidates.columns:
            exec_time_col = "user_attrs_computation_time_seconds"
        elif "computation_time_seconds" in best_candidates.columns:
            exec_time_col = "computation_time_seconds"
        
        if exec_time_col:
            best_row = best_candidates.loc[best_candidates[exec_time_col].astype(float).idxmin()]
        else:
            best_row = best_candidates.iloc[0]
    else:
        best_row = trials_df.iloc[0]

    best_params = {}
    for column, value in best_row.items():
        if not column.startswith("params_"):
            continue
        if pd.isna(value):
            continue
        best_params[column.removeprefix("params_")] = value

    user_attrs = {}
    for column, value in best_row.items():
        if not column.startswith("user_attrs_"):
            continue
        if pd.isna(value):
            continue
        user_attrs[column.removeprefix("user_attrs_")] = value

    best_value = float(best_row["value"]) if "value" in best_row else None
    return best_params, user_attrs, best_value, trials_csv


def resolve_training_epochs(user_attrs):
    if "best_epoch" in user_attrs:
        return int(user_attrs["best_epoch"])
    if "trained_epochs" in user_attrs:
        return int(user_attrs["trained_epochs"])
    return int(DEFAULT_TUNING_MAX_EPOCHS)


def build_model_config(model_name, best_params):
    if model_name not in BUILD_CONFIGS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {list(BUILD_CONFIGS.keys())}"
        )

    params = apply_architecture_preset(model_name, dict(best_params))
    if "win_size" not in params:
        raise ValueError(
            f"Missing 'win_size' in best parameters for model '{model_name}'")

    return BUILD_CONFIGS[model_name](params), params


def _to_seed_list(seeds_value):
    if seeds_value is None:
        return []
    if isinstance(seeds_value, int):
        return [seeds_value]
    if isinstance(seeds_value, (list, tuple)):
        return [int(seed) for seed in seeds_value]
    if isinstance(seeds_value, str):
        return [int(seed.strip()) for seed in seeds_value.split(",") if seed.strip()]
    raise ValueError(f"Unsupported seeds format: {type(seeds_value)}")


def run_evaluation(model_name, best_by_strategy, dataset_path, file_list, seeds, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class = MODEL_SPECS[model_name]["model_class"]
    evaluation_start_time = time.perf_counter()

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Evaluate both strategies
    strategies = ["overlapping"]
    strategy_results = {}

    for strategy in strategies:
        strategy_best = best_by_strategy[strategy]
        best_params = strategy_best["best_params"]
        user_attrs = strategy_best["user_attrs"]
        best_value = strategy_best["best_value"]
        trials_csv = strategy_best["trials_csv"]

        config, resolved_params = build_model_config(model_name, best_params)

        training_epochs = resolve_training_epochs(user_attrs)
        # print(best_params)
        # print(f"Resolved training epochs: {training_epochs}")
        # import sys
        # sys.exit(0)
        use_scheduler = resolved_params["lr_mode"] == "scheduled"
        scheduler_name = DEFAULT_SCHEDULED_LR_SCHEDULER if use_scheduler else "none"
        scheduler_kwargs = (
            scheduled_lr_kwargs(
                training_epochs,
                scheduler_name,
                DEFAULT_SCHEDULED_LR_SCHEDULER_KWARGS,
            )
            if use_scheduler
            else {}
        )

        seed_frames = []
        for seed in seeds:
            set_seed(seed)

            trainer = Trainer(
                batch_size=DEFAULT_BATCH_SIZE,
                lr=float(resolved_params["lr"]),
                device=device,
                win_size=int(resolved_params["win_size"]),
                validation_size=DEFAULT_VALIDATION_SIZE,
                lr_scheduler=scheduler_name,
                lr_scheduler_kwargs=scheduler_kwargs,
            )

            evaluator = Evaluator(
                batch_size=DEFAULT_EVAL_BATCH_SIZE,
                device=device,
                metrics="restr",
                strategy=strategy,
            )

            rows = []
            for filename in tqdm(file_list, desc=f"{model_name} {strategy} seed={seed}"):
                data_train, data, labels = load_dataset(
                    dataset_path, filename)

                rank = utils.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

                model = model_class(config).to(device)
                start_time = time.time()
                trainer.train(model, data_train, training_epochs)
                reconstruction_error = evaluator.reconstruction_error(
                                data, model, int(resolved_params["win_size"]), 1)
                end_time = time.time()
                metrics = evaluator.metrics_fnc(
                                reconstruction_error, labels, slidingWindow=rank)
     

                row = {
                    "filename": filename,
                    "seed": seed,
                    "execution_time_seconds": end_time - start_time,
                }
                row.update(metrics)
                rows.append(row)

            seed_df = pd.DataFrame(rows)
            seed_df.to_csv(os.path.join(model_output_dir,
                           f"seed{seed}_{strategy}.csv"), index=False)
            seed_frames.append(seed_df)
            print(seed_df.mean(numeric_only=True))

        combined_df = pd.concat(seed_frames, ignore_index=True)
        mean_df = (
            combined_df.groupby("filename", as_index=False)
            .mean(numeric_only=True)
            .sort_values("filename")
        )
        std_df = (
            combined_df.groupby("filename", as_index=False)
            .std(numeric_only=True)
            .sort_values("filename")
        )

        mean_df.to_csv(os.path.join(model_output_dir, f"mean_{strategy}.csv"), index=False)
        std_df.to_csv(os.path.join(model_output_dir, f"std_{strategy}.csv"), index=False)

        strategy_results[strategy] = {
            "mean_df": mean_df,
            "std_df": std_df,
            "summary": mean_df.mean(numeric_only=True).to_dict(),
            "resolved_params": resolved_params,
            "training_epochs": training_epochs,
            "scheduler": scheduler_name,
            "best_value": best_value,
            "trials_csv": trials_csv,
        }

    # Create summary rows for both strategies
    summary_rows = []
    for strategy in strategies:
        strategy_result = strategy_results[strategy]
        summary = strategy_result["summary"]
        resolved_params = strategy_result["resolved_params"]
        summary_row = {
            "model": model_name,
            "strategy": strategy,
            "trials_csv": strategy_result["trials_csv"],
            "best_win_size": int(resolved_params["win_size"]),
            "lr": float(resolved_params["lr"]),
            "architecture_size": str(resolved_params["architecture_size"]),
            "lr_mode": str(resolved_params["lr_mode"]),
            "scheduler": strategy_result["scheduler"],
            "trained_epochs": int(strategy_result["training_epochs"]),
            "total_execution_time_seconds": float(time.perf_counter() - evaluation_start_time),
            "best_trial_value": strategy_result["best_value"],
            "AUC-PR": summary.get("AUC-PR", np.nan),
            "AUC-ROC": summary.get("AUC-ROC", np.nan),
            "Standard-F1": summary.get("Standard-F1", np.nan),
        }
        summary_rows.append(summary_row)

    return strategy_results, summary_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate each model using the best hyperparameters from tuning results."
    )
    parser.add_argument(
        "--target-models",
        nargs="+",
        default=DEFAULT_TARGET_MODELS,
        help="Models to evaluate. Example: --target-models AutoEncoder TimesNet",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--file-list",
        default=DEFAULT_FILE_LIST,
        help="CSV file containing the file_name column for evaluation.",
    )
    parser.add_argument(
        "--tuning-results-dir",
        default=DEFAULT_TUNING_RESULTS_DIR,
        help="Directory containing the tuning CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation results will be saved.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds used for repeated evaluation runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    unknown_models = [
        model_name for model_name in args.target_models if model_name not in MODEL_SPECS
    ]
    if unknown_models:
        raise ValueError(
            f"Unknown models in target_models: {unknown_models}. "
            f"Available models: {list(MODEL_SPECS.keys())}"
        )

    file_list = pd.read_csv(args.file_list)["file_name"].values
    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    for model_name in tqdm(args.target_models, desc="Models"):
        best_by_strategy = {}
        for strategy in ["overlapping", "disjoint"]:
            best_params, user_attrs, best_value, trials_csv = load_best_params(
                args.tuning_results_dir,
                model_name,
                strategy=strategy,
            )
            best_by_strategy[strategy] = {
                "best_params": best_params,
                "user_attrs": user_attrs,
                "best_value": best_value,
                "trials_csv": trials_csv,
            }

        strategy_results, model_summary_rows = run_evaluation(
            model_name,
            best_by_strategy,
            args.dataset_path,
            file_list,
            args.seeds,
            args.output_dir,
        )

        summary_rows.extend(model_summary_rows)

        for strategy in ["overlapping", "disjoint"]:
            strategy_best = best_by_strategy[strategy]
            print(
                f"[{model_name}] strategy={strategy} best trial value: {strategy_best['best_value']}"
            )
            print(f"[{model_name}] strategy={strategy} best params: {strategy_best['best_params']}")
        print(f"[{model_name}] results saved in {os.path.join(args.output_dir, model_name)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(
        args.output_dir, "summary.csv"), index=False)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    print(summary_df.to_string())
    print("="*80)
    print(f"\nResults saved to: {os.path.join(args.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
