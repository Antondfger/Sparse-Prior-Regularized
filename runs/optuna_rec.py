"""
Run experiment.
"""

import os
import sys
import time

sys.path.append(os.environ["PATH4SEQ"])
os.environ["WORLD_SIZE"] = "0"

import random

import hydra
import pandas as pd
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nn.metrics import Evaluator
from preprocessing.preparation import get_last_item, remove_last_item
from preprocessing.preprocessing import drop_short_sequences
from runs.dl import create_dataloaders, create_model, predict, training


@hydra.main(config_path="conf", config_name="SASRec_optuna")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)

    if hasattr(config, "project_name"):
        task = Task.init(
            project_name=config.project_name,
            task_name=config.task_name,
            reuse_last_task_id=False,
        )
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    path_to_split = config.datasets_info.path_to_split_data

    train = pd.read_csv(path_to_split + "train_" + config.datasets_info.name + ".csv")

    validation = pd.read_csv(
        path_to_split + "validation_" + config.datasets_info.name + ".csv"
    )
    max_item_id = int(max(train.item_id.max(), validation.item_id.max()))

    train = drop_short_sequences(train, 2)
    train["item_id"] = train["item_id"].astype(int)
    validation = drop_short_sequences(validation, 3)
    validation["item_id"] = validation["item_id"].astype(int)
    validation_gt = get_last_item(validation)
    validation_inputs = remove_last_item(validation)

    seed = config.random_state
    random.seed(seed)
    if torch.cuda.is_available():
        seed_everything(seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_loader, eval_loader = create_dataloaders(train, validation_inputs, config)
    model = create_model(config, item_count=max_item_id)
    time.time()
    trainer, seqrec_module, model = training(
        model, train_loader, eval_loader, train, max_item_id, config
    )

    recs_validation = predict(trainer, seqrec_module, validation_inputs, config)

    val_metrics = evaluate(
        recs_validation, validation_gt, train, task, config, prefix="val"
    )

    print(val_metrics)
    return val_metrics


def evaluate(recs, test_last, train, task, config, prefix="test"):

    all_metrics = {}

    for k in config.top_k_metrics:
        evaluator = Evaluator(topk=[k])
        metrics = evaluator.compute_metrics(test_last, recs, train)
        metrics = {prefix + "_" + key: value for key, value in metrics.items()}
        all_metrics.update(metrics)

    print(all_metrics)
    if task:

        clearml_logger = task.get_logger()

        for key, value in all_metrics.items():
            clearml_logger.report_single_value(key, value.item())
            if str(key) == "val_" + config.optimization_metric:
                val_metrics = value.item()

    return float(val_metrics)


if __name__ == "__main__":

    main()
