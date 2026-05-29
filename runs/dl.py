"""
Run experiment.
"""
import os
import time
import sys
sys.path.append(os.environ['PATH4SEQ'])

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from nn.datasets import (CausalLMDataset, CausalLMPredictionDataset,
                         PaddingCollateFn)
from nn.metrics import Evaluator
from nn.models import SASRec, GRU4Rec
from nn.modules import SeqRec, SeqRecWithSampling
from nn.postprocess import preds2recs
from preprocessing.preparation import get_last_item, remove_last_item
from preprocessing.preprocessing import preprocessing, drop_short_sequences
from preprocessing.splitter import session_split
from replay.metrics import Surprisal
from entmax import entmax15
import torch
import random

@hydra.main(config_path="conf", config_name="SASRec")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if hasattr(config, 'project_name'):
        task = Task.init(project_name=config.project_name,
                         task_name=config.task_name,
                         reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    seed = config.random_state
    random.seed(seed)
    if torch.cuda.is_available():
        seed_everything(seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    path_to_split = config.datasets_info.path_to_split_data

    train = pd.read_csv(path_to_split + 'train_' +
                            config.datasets_info.name + '.csv')
        
    test = pd.read_csv(path_to_split + 'test_' +
                        config.datasets_info.name + '.csv')
    
    validation = pd.read_csv(path_to_split + 'validation_' +
                                config.datasets_info.name + '.csv')
    max_item_id = max(train.item_id.max(), test.item_id.max(),
                        validation.item_id.max())
    
    train = prepare_dataset(train)
    validation = prepare_dataset(validation)
    test = prepare_dataset(test)

    counts_tensor = prepare_item_counts(
        train, int(max_item_id + 1), counts=config.counts
    )

    train_loader, eval_loader = create_dataloaders(train, validation, config, counts_tensor)
    model = create_model(config, item_count=int(max_item_id))
    start_time = time.time()
    trainer, seqrec_module, model = training(model, train_loader, eval_loader, 
                                             train, max_item_id, config)
    training_time = time.time() - start_time
    print('training_time', training_time)

    val_inputs = remove_last_item(validation)
    val_last_item = get_last_item(validation)
    recs_validation = predict(trainer, seqrec_module, val_inputs, config)
    
    if task is not None:
        task.upload_artifact(name="val_recs", artifact_object=recs_validation)
        
    evaluate(recs_validation, val_last_item, train, task, config, prefix='val')

    test_inputs = remove_last_item(test)
    test_last_item = get_last_item(test)

    test_inputs = filter_cold(test_inputs, train)

    recs_test = predict(trainer, seqrec_module, test_inputs, config)
    
    if task is not None:
        task.upload_artifact(name="test_recs", artifact_object=recs_test)
    
    evaluate(recs_test, test_last_item, train, task, config, prefix=f'test_fs_{seqrec_module.filter_seen}')

def prepare_dataset(dataset):
    dataset = drop_short_sequences(dataset, 2)
    dataset['item_id'] = dataset['item_id'].astype(int)
    return dataset


def create_dataloaders(train, validation, config, counts_tensor):

    train_dataset = CausalLMDataset(
        train,
        item_counts_tensor=counts_tensor,
        **config["dataset_params"]
    )
    eval_dataset = CausalLMPredictionDataset(
        validation,
        max_length=config.dataset_params.max_length,
        validation_mode=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        collate_fn=PaddingCollateFn(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataloader.test_batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        collate_fn=PaddingCollateFn(),
    )

    return train_loader, eval_loader


def create_model(config, item_count):

    if hasattr(config.dataset_params, 'num_negatives') and config.dataset_params.num_negatives:
        add_head = False
    else:
        add_head = True

    if config.model == 'SASRec':
        model = SASRec(item_num=item_count, add_head=add_head, **config.model_params)

    if config.model=='GRU4Rec':
        model = GRU4Rec(vocab_size=item_count + 1, add_head=add_head, rnn_config=config.model_params)


    return model


def training(model, train_loader, eval_loader, train, max_item_id, config):
    counts_tensor = prepare_item_counts(train, int(max_item_id+1), counts=config.counts)
    print(counts_tensor) 
    
    if config.dataset_params.num_negatives is not None:
        seqrec_module = SeqRecWithSampling(model, **config['seqrec_module'], temp=config.temp, 
                                           type_loss=config.type_loss, 
                                           gamma=config.gamma, item_counts=counts_tensor)

    else:
        seqrec_module = SeqRec(model, **config['seqrec_module'], temp=config.temp, 
                                     gamma=config.gamma, item_counts=counts_tensor, type_loss=config.type_loss)
        
    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True,
                         **config['trainer_params'])

    trainer.fit(model=seqrec_module,
            train_dataloaders=train_loader,
            val_dataloaders=eval_loader)

    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, seqrec_module, model


def predict(trainer, seqrec_module, data, config):

    predict_dataset = CausalLMPredictionDataset(
        data, max_length=config.dataset_params.max_length)

    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=config.dataloader.test_batch_size,
        num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.top_k_metrics)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)

    recs = preds2recs(preds)
    print('recs shape', recs.shape)

    return recs


def evaluate(recs, test_last, train, task, config, prefix='test'):

    all_metrics = {}

    for k in config.top_k_metrics:
        evaluator = Evaluator(topk=[k])
        metrics = evaluator.compute_metrics(test_last, recs, train)
        metrics = {prefix + '_' + key: value for key, value in metrics.items()}
        all_metrics.update(metrics)

    print(all_metrics)
    if task:

        clearml_logger = task.get_logger()

        for key, value in all_metrics.items():
            clearml_logger.report_single_value(key, value.item())
        all_metrics = pd.Series(all_metrics).to_frame().reset_index()
        all_metrics.columns = ['metric_name', 'metric_value']
    
    return metrics

def filter_cold(test, train):
    train_items = train['item_id'].unique()

    test = test[test['item_id'].isin(train_items)]

    return drop_short_sequences(test, 2)

def prepare_item_counts(train_df, n_classes, counts=False):
    counts_map = train_df['item_id'].value_counts()
    
    counts_tensor = torch.ones(n_classes)
    
    indices = torch.tensor(counts_map.index.values, dtype=torch.long)
    values = torch.tensor(counts_map.values, dtype=torch.float)
        
    if counts:
        counts_tensor[indices] = values

    else:
       counts_tensor[indices] = values / len(train_df['item_id'])
        
    return counts_tensor

if __name__ == "__main__":

    main()