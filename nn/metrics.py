"""
Metrics.
"""

import pandas as pd
import numpy as np
import torch
from replay import metrics as replay_base_class
from replay.metrics import OfflineMetrics

from tqdm.auto import tqdm

REPLAY_METRICS = ["NDCG", "HitRate", "Surprisal", "Novelty", "Coverage"]

class Evaluator:
    """Class for computing recommendation metrics using Replay and RecBole."""

    def __init__(
        self,
        replay_metrics=REPLAY_METRICS,
        topk=[10, 100],
        modes=["Mean"],
        user_id="user_id",
        item_id="item_id",
        rating_columns="prediction",
    ):
        self.replay_metrics = replay_metrics
        self.topk = sorted(topk)
        self.modes = modes
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating_columns

    def compute_metrics(self, test, recs, train=None, base_recommendations=None):
        """Compute all metrics from Replay and RecBole."""
        
        metrics_list = []
        for metric in tqdm(self.replay_metrics, desc='replay metrics'):
            for k in self.topk:
                if hasattr(replay_base_class, 'mode'):
                    for mode in self.modes:
                        mode_instance = getattr(replay_base_class, mode)()
                        metrics_list.append(
                            getattr(replay_base_class, metric)(topk=k, mode=mode_instance)
                        )
                else:
                    metrics_list.append(getattr(replay_base_class, metric)(topk=k))

        results = OfflineMetrics(
            metrics_list,
            query_column=self.user_id,
            item_column=self.item_id,
            rating_column=self.rating,
        )(recs, test, train, base_recommendations)
        
        metrics_df = pd.DataFrame.from_dict(results, orient="index").T

        return metrics_df