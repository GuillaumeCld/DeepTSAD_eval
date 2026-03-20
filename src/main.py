"""
Run the script with:
    python src/main.py --config configs/experiment.conf --model_config configs/TimesNet.conf
"""

from eval import Evaluator
from training import Trainer
import pandas as pd
import models
from tqdm import tqdm

import numpy as np
import torch
import random
from procedure import train_and_evaluate

import configparser
from types import SimpleNamespace
import argparse
import os
import importlib

def load_model(models_pkg, name):
    module = importlib.import_module(f"{models_pkg}.{name}")
    return module.Model

def load_config(path: str) -> SimpleNamespace:

    parser = configparser.ConfigParser()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.lstrip().startswith("["):
        content = "[DEFAULT]\n" + content
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


def main(config_path: str, model_config_path: str):

    config = load_config(config_path)
    model_config = load_config(model_config_path)
    model_config.seq_len = config.win_size
    model_config.label_len = config.win_size

 
    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    out_dir = os.path.join("results", str(model_config.model_name))
    os.makedirs(out_dir, exist_ok=True)


    trainer = Trainer(
        batch_size=config.batch_size,
        lr=config.lr,
        device=config.device,
        win_size=config.win_size,
        validation_size=config.validation_size,
    )

    evaluator = Evaluator(
        batch_size=config.batch_size,
        device=config.device,
        metrics=config.metrics,
        strategy=config.strategy
    )

    results = []
    results_df = pd.DataFrame()  

    file_list = pd.read_csv(config.file_list)['file_name'].values
    for filename in tqdm(file_list):
        
        # Select model via name from config
        ModelClass = load_model("models", model_config.model_name)
        model = ModelClass(model_config)
       
        metrics = train_and_evaluate(
            config.path,
            filename,
            model,
            trainer,
            evaluator,
            win_size=config.win_size,
            epochs=config.epochs
        )

        result = {"filename": filename}
        result.update(metrics)
        results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(out_dir, f"32_{config.seed}.csv"),
            index=False
        )

    if len(results_df) == 0:
        print("No files processed (file_list was empty).")
    else:
        print((results_df.mean(numeric_only=True) * 100).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Path to flat training/eval .conf file")
    ap.add_argument("--model_config", required=True,
                    help="Path to flat model .conf file")
    args = ap.parse_args()

    main(args.config, args.model_config)
