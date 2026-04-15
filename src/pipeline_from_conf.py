"""Generic pipeline runner driven by one .conf file.

Expected format:
    [experiment]
    ...
    [model]
    ...
"""

from __future__ import annotations

import argparse
import ast
import configparser
import importlib
import os
import random
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from eval import Evaluator
from procedure import train_and_evaluate
from training import Trainer


def _parse_value(raw_value: str):
    value = raw_value.strip()
    if value == "":
        return ""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_sections(config_path: str) -> tuple[Dict[str, object], Dict[str, object]]:
    parser = configparser.ConfigParser()
    parser.read(config_path)

    if "experiment" not in parser or "model" not in parser:
        raise ValueError("Config must contain [experiment] and [model] sections")

    experiment = {k: _parse_value(v) for k, v in parser["experiment"].items()}
    model = {k: _parse_value(v) for k, v in parser["model"].items()}
    return experiment, model


def _to_seed_list(seeds_value) -> List[int]:
    if seeds_value is None:
        return []
    if isinstance(seeds_value, int):
        return [seeds_value]
    if isinstance(seeds_value, (list, tuple)):
        return [int(s) for s in seeds_value]
    if isinstance(seeds_value, str):
        return [int(s.strip()) for s in seeds_value.split(",") if s.strip()]
    raise ValueError(f"Unsupported seeds format: {type(seeds_value)}")


def _require(config: Dict[str, object], keys: List[str], section: str):
    for key in keys:
        if key not in config:
            raise ValueError(f"Missing '{key}' in [{section}] section")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _build_model(model_conf: Dict[str, object]):
    _require(model_conf, ["model_name"], "model")

    module = importlib.import_module(f"models.{model_conf['model_name']}")
    model_args = dict(model_conf)
    return module.Model(SimpleNamespace(**model_args))


def run_pipeline(config_path: str):
    exp_conf, model_conf = load_sections(config_path)

    _require(
        exp_conf,
        [
            "device",
            "path",
            "file_list",
            "batch_size",
            "lr",
            "epochs",
            "win_size",
            "validation_size",
            "metrics",
            "strategy",
        ],
        "experiment",
    )

    win_size = int(exp_conf["win_size"])
    model_conf.setdefault("seq_len", win_size)
    model_conf.setdefault("label_len", win_size)

    seeds = _to_seed_list(exp_conf.get("seeds"))
    if not seeds:
        seeds = [int(exp_conf.get("seed", 42))]

    file_list = pd.read_csv(str(exp_conf["file_list"]))["file_name"].values

    model_name = str(model_conf["model_name"])
    output_dir = str(exp_conf.get("output_dir", os.path.join("results", model_name)))
    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        _set_seed(seed)

        trainer = Trainer(
            batch_size=int(exp_conf["batch_size"]),
            lr=float(exp_conf["lr"]),
            device=str(exp_conf["device"]),
            win_size=win_size,
            validation_size=float(exp_conf["validation_size"]),
        )

        evaluator = Evaluator(
            batch_size=int(exp_conf["batch_size"]),
            device=str(exp_conf["device"]),
            metrics=str(exp_conf["metrics"]),
            strategy=str(exp_conf["strategy"]),
        )

        rows = []
        for filename in tqdm(file_list, desc=f"{model_name} seed={seed}"):
            model = _build_model(model_conf)
            metrics = train_and_evaluate(
                str(exp_conf["path"]),
                filename,
                model,
                trainer,
                evaluator,
                win_size=win_size,
                epochs=int(exp_conf["epochs"]),
                stride=int(exp_conf.get("stride", 1)),
            )
            row = {"filename": filename}
            row.update(metrics)
            rows.append(row)

        result_df = pd.DataFrame(rows)
        output_name = exp_conf.get("output_name")
        if output_name:
            output_file = str(output_name).format(seed=seed, model=model_name, ws=win_size)
        else:
            output_file = f"ws{win_size}_seed{seed}.csv"

        save_path = os.path.join(output_dir, output_file)
        result_df.to_csv(save_path, index=False)

        if len(result_df) == 0:
            print(f"[{model_name}][seed={seed}] No files processed.")
        else:
            print(f"[{model_name}][seed={seed}] {save_path}")
            print((result_df.mean(numeric_only=True) * 100).round(3))


def main():
    parser = argparse.ArgumentParser(description="Run generic anomaly pipeline from .conf")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a .conf file with [experiment] and [model] sections",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
