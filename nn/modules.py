"""
Pytorch Lightning Modules.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from entmax import entmax15_loss
from torch import nn


class SeqRecBase(pl.LightningModule):

    def __init__(
        self, model, lr=1e-3, padding_idx=0, predict_top_k=10, filter_seen=True
    ):

        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.filter_seen = filter_seen

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)

        scores = scores.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        user_ids = batch["user_id"].detach().cpu().numpy()

        return {"preds": preds, "scores": scores, "user_ids": user_ids}

    def validation_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)
        metrics = self.compute_val_metrics(batch["target"], preds)

        self.log("val_ndcg", metrics["ndcg"], prog_bar=True)
        self.log("val_hit_rate", metrics["hit_rate"], prog_bar=True)
        self.log("val_mrr", metrics["mrr"], prog_bar=True)

    def make_prediction(self, batch):

        outputs = self.prediction_output(batch)

        input_ids = batch["input_ids"]
        rows_ids = torch.arange(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1
        preds = outputs[rows_ids, last_item_idx, :]

        scores, preds = torch.sort(preds, descending=True)

        if self.filter_seen:
            seen_items = batch["full_history"]
            preds, scores = self.filter_seen_items(preds, scores, seen_items)
        else:
            scores = scores[:, : self.predict_top_k]
            preds = preds[:, : self.predict_top_k]

        return preds, scores

    def filter_seen_items(self, preds, scores, seen_items):

        max_len = seen_items.size(1)
        scores = scores[:, : self.predict_top_k + max_len]
        preds = preds[:, : self.predict_top_k + max_len]

        final_preds, final_scores = [], []
        for i in range(preds.size(0)):
            not_seen_indexes = torch.isin(preds[i], seen_items[i], invert=True)
            pred = preds[i, not_seen_indexes][: self.predict_top_k]
            score = scores[i, not_seen_indexes][: self.predict_top_k]
            final_preds.append(pred)
            final_scores.append(score)

        final_preds = torch.vstack(final_preds)
        final_scores = torch.vstack(final_scores)

        return final_preds, final_scores

    def compute_val_metrics(self, targets, preds):

        ndcg, hit_rate, mrr = 0, 0, 0

        for i, pred in enumerate(preds):
            if torch.isin(targets[i], pred).item():
                hit_rate += 1
                rank = torch.where(pred == targets[i])[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
                mrr += 1 / rank

        hit_rate = hit_rate / len(targets)
        ndcg = ndcg / len(targets)
        mrr = mrr / len(targets)

        return {"ndcg": ndcg, "hit_rate": hit_rate, "mrr": mrr}


class SeqRec(SeqRecBase):
    def __init__(
        self,
        model,
        lr=1e-3,
        padding_idx=0,
        predict_top_k=10,
        filter_seen=True,
        type_loss="ce",
        temp=1,
        tau_adj=None,
        type_adj=None,
        item_counts=None,
        decay_epochs=5,
        margin_type="log",
    ):

        super().__init__(model, lr, padding_idx, predict_top_k, filter_seen)
        self.tau_adj = tau_adj
        self.type_adj = type_adj
        self.type_loss = type_loss
        self.n_classes = len(item_counts)
        self.decay_epochs = decay_epochs
        self.temp = temp
        self.margin_type = margin_type

        if hasattr(self.model, "item_emb"):  # for SASRec
            self.embed_layer = self.model.item_emb
        elif hasattr(self.model, "embed_layer"):  # for other models
            self.embed_layer = self.model.embed_layer

        if self.tau_adj is not None:
            if self.type_adj == "logit":
                print("logit")
                item_counts = torch.log(item_counts) * self.tau_adj

            elif self.type_adj == "margin":
                print("margin")
                item_counts = -1 / (item_counts * self.tau_adj) ** 0.25

            elif self.type_adj == "rps":
                print("rps")
                item_counts = (1 / (item_counts + 10e-7) ** self.tau_adj) * 0.005

            elif self.type_adj in ["bc_cos", "bc_angle", "bc_dot"]:
                print(self.type_adj)
                if self.margin_type == "rank":
                    print("margin_type: rank")
                    # Ранги: 0 для самого популярного, N-1 для наименее популярного
                    item_ranks = item_counts.argsort(descending=True).argsort()
                    # Инвертируем, чтобы у самого популярного был 1.0, а у редкого 0.0
                    normalized_pop = 1.0 - (item_ranks.float() / (self.n_classes - 1))
                    item_counts = normalized_pop * self.tau_adj
                else:
                    print("margin_type: log")
                    log_counts = torch.log(item_counts)
                    # Нормализуем логарифм популярности так, чтобы максимум был равен 1.
                    # Это делает параметр tau_adj предсказуемым (максимальный бонус)
                    # и независимым от размера датасета.
                    max_log_count = torch.max(log_counts)
                    item_counts = (log_counts / max_log_count) * self.tau_adj
                
                if self.type_adj == "bc_angle":
                    # Ограничиваем марджин, чтобы он строго был меньше pi
                    item_counts = torch.clamp(item_counts, max=torch.pi - 0.01)

            self.register_buffer("counts", item_counts)
            print(self.counts)

        if self.type_loss == "ce" and self.type_adj == "rps":
            self.loss = nn.CrossEntropyLoss(weight=self.counts)

        elif self.type_loss == "ce":
            self.loss = nn.CrossEntropyLoss(reduction="none")

        elif self.type_loss == "entmax":
            self.loss = entmax15_loss

    def training_step(self, batch, batch_idx):

        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, outputs, batch):

        labels = batch["labels"].view(-1)

        if self.type_adj in ["bc_cos", "bc_angle"]:
            outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            weights_norm = torch.nn.functional.normalize(self.embed_layer.weight, p=2, dim=-1)
            logits = torch.matmul(outputs_norm, weights_norm.T)
            logits = logits.view(-1, logits.size(-1))
            
            valid_mask = (labels != self.padding_idx) & (labels != -100)
            valid_labels = labels[valid_mask]
            
            if self.type_adj == "bc_cos":
                margin = self.counts[valid_labels]
                # Ограничиваем логит снизу, чтобы он не ушел далеко за -1
                logits[valid_mask, valid_labels] = torch.clamp(logits[valid_mask, valid_labels] + margin, min=-1.0)
            elif self.type_adj == "bc_angle":
                cos_theta = logits[valid_mask, valid_labels]
                margin = self.counts[valid_labels]
                # cos(theta + margin) = cos(theta)cos(margin) - sin(theta)sin(margin)
                sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=1e-7))
                cos_theta_m = cos_theta * torch.cos(margin) - sin_theta * torch.sin(margin)
                
                # ArcFace trick: защита от осцилляции при theta + margin > pi
                cond = cos_theta > -torch.cos(margin)
                # Возвращаем стабильный fallback (как в оригинальном InsightFace)
                fallback = cos_theta - margin * torch.sin(margin)
                logits[valid_mask, valid_labels] = torch.where(cond, cos_theta_m, fallback)
        elif self.type_adj == "bc_dot":
            logits = outputs.view(-1, outputs.size(-1))
            valid_mask = (labels != self.padding_idx) & (labels != -100)
            valid_labels = labels[valid_mask]
            logits[valid_mask, valid_labels] += self.counts[valid_labels]
        else:
            logits = outputs.view(-1, outputs.size(-1))
            if self.tau_adj is not None and self.type_adj != "rps":
                logits = logits + self.counts

        if self.type_loss == "ce":
            loss = self.loss(logits / self.temp, labels)

        if self.type_loss == "entmax":
            mask = (labels != self.padding_idx) & (labels != -100)
            logits = logits[mask]
            labels = labels[mask]

            loss = self.loss(logits / self.temp, labels, k=self.get_current_k())

        return loss.mean()

    def prediction_output(self, batch):

        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        if outputs.size(-1) != self.n_classes:
            if self.type_adj in ["bc_cos", "bc_angle"]:
                outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
                weights_norm = torch.nn.functional.normalize(self.embed_layer.weight, p=2, dim=-1)
                outputs = torch.matmul(outputs_norm, weights_norm.T) / self.temp
            else:
                outputs = torch.matmul(outputs, self.embed_layer.weight.T) / self.temp
        return outputs

    def get_current_k(self):
        max_k = self.n_classes
        min_k = int(0.005 * self.n_classes)
        if self.current_epoch >= self.decay_epochs:
            return min_k
        else:
            decay_factor = 0.5 ** (self.current_epoch - 1)
            return int(min_k + max_k * decay_factor)


class SeqRecWithSampling(SeqRec):

    def __init__(
        self,
        model,
        lr=1e-3,
        type_loss="ce",
        padding_idx=0,
        predict_top_k=10,
        filter_seen=True,
        temp=10,
        decay_epochs=5,
        tau_adj=None,
        type_adj=None,
        item_counts=None,
        margin_type="log",
    ):

        super().__init__(
            model,
            lr,
            padding_idx,
            predict_top_k,
            filter_seen,
            type_loss,
            temp,
            tau_adj,
            type_adj,
            item_counts,
            decay_epochs,
            margin_type,
        )

    def compute_loss(self, outputs, batch):

        if batch["negatives"].ndim == 2:  # for full_negative_sampling=False
            # [N, M, D]
            embeds_negatives = self.embed_layer(batch["negatives"].to(torch.int32))
            if self.type_adj in ["bc_cos", "bc_angle"]:
                outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
                embeds_negatives_norm = torch.nn.functional.normalize(embeds_negatives, p=2, dim=-1)
                logits_negatives = torch.matmul(outputs_norm, embeds_negatives_norm.transpose(1, 2))
            else:
                # [N, T, D] * [N, D, M] -> [N, T, M]
                logits_negatives = torch.matmul(outputs, embeds_negatives.transpose(1, 2))

            neg_ids = batch["negatives"].unsqueeze(1)

        elif batch["negatives"].ndim == 3:  # for full_negative_sampling=True
            # [N, T, M, D]
            embeds_negatives = self.embed_layer(batch["negatives"].to(torch.int32))
            if self.type_adj in ["bc_cos", "bc_angle"]:
                outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
                embeds_negatives_norm = torch.nn.functional.normalize(embeds_negatives, p=2, dim=-1)
                logits_negatives = torch.matmul(
                    outputs_norm.unsqueeze(2), embeds_negatives_norm.transpose(2, 3)
                ).squeeze()
            else:
                # [N, T, 1, D] * [N, T, D, M] -> [N, T, 1, M] -> -> [N, T, M]
                logits_negatives = torch.matmul(
                    outputs.unsqueeze(2), embeds_negatives.transpose(2, 3)
                ).squeeze()
            neg_ids = batch["negatives"]

            if logits_negatives.ndim == 2:
                logits_negatives = logits_negatives.unsqueeze(2)

        # embed  and compute logits for positives
        # [N, T]
        labels = batch["labels"].clone()
        labels[labels == -100] = self.padding_idx
        # [N, T, D]
        embeds_labels = self.embed_layer(labels)
        if self.type_adj in ["bc_cos", "bc_angle"]:
            outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            embeds_labels_norm = torch.nn.functional.normalize(embeds_labels, p=2, dim=-1)
            logits_labels = torch.matmul(
                outputs_norm.unsqueeze(2), embeds_labels_norm.unsqueeze(3)
            ).squeeze()
        else:
            # [N, T, 1, D] * [N, T, D, 1] -> [N, T, 1, 1] -> [N, T]
            logits_labels = torch.matmul(
                outputs.unsqueeze(2), embeds_labels.unsqueeze(3)
            ).squeeze()

        # concat positives and negatives
        # [N, T, M + 1]
        logits = torch.cat([logits_labels.unsqueeze(2), logits_negatives], dim=-1)

        # prepare targets for loss
        if self.type_loss == "ce" or self.type_loss == "entmax":
            # [N, T]
            targets = batch["labels"].clone()
            targets[targets != -100] = 0

        elif self.type_loss == "bce":
            # [N, T, M + 1]
            targets = torch.zeros_like(logits)
            targets[:, :, 0] = 1

        if self.tau_adj is not None and (
            self.type_adj == "logit" or self.type_adj == "margin"
        ):
            pos_ids = labels
            neg_ids = batch["negatives"]

            if neg_ids.ndim == 2:
                neg_ids = neg_ids.unsqueeze(1).expand(-1, pos_ids.size(1), -1)

            all_sampled_ids = torch.cat([pos_ids.unsqueeze(2), neg_ids], dim=-1)
            sampled_counts = self.counts[all_sampled_ids.to(torch.long)]

            logits = logits + sampled_counts

        if self.tau_adj is not None and self.type_adj == "bc_dot":
            pos_ids = labels
            pos_counts = self.counts[pos_ids.to(torch.long)]
            logits[:, :, 0] = logits[:, :, 0] + pos_counts
            
        elif self.tau_adj is not None and self.type_adj == "bc_cos":
            pos_ids = labels
            pos_counts = self.counts[pos_ids.to(torch.long)]
            logits[:, :, 0] = torch.clamp(logits[:, :, 0] + pos_counts, min=-1.0)

        elif self.tau_adj is not None and self.type_adj == "bc_angle":
            pos_ids = labels
            margin = self.counts[pos_ids.to(torch.long)]
            cos_theta = logits[:, :, 0]
            sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=1e-7))
            cos_theta_m = cos_theta * torch.cos(margin) - sin_theta * torch.sin(margin)
            
            # ArcFace trick: защита от осцилляции при theta + margin > pi
            cond = cos_theta > -torch.cos(margin)
            fallback = cos_theta - margin * torch.sin(margin)
            logits[:, :, 0] = torch.where(cond, cos_theta_m, fallback)

        if self.type_loss == "entmax":

            logits = logits / self.temp

            mask = (targets != -100).view(-1)
            valid_logits = logits.view(-1, logits.size(-1))[mask]
            valid_targets = targets.view(-1)[mask]

            loss = entmax15_loss(valid_logits, valid_targets, k=self.get_current_k())

        elif self.type_loss == "ce":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))

        if self.tau_adj is not None and self.type_adj == "rps":
            labels = batch["labels"].view(-1)
            loss = loss * self.counts[labels]

        return loss.mean()

    def prediction_output(self, batch):

        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        if self.type_adj in ["bc_cos", "bc_angle"]:
            outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            weights_norm = torch.nn.functional.normalize(self.embed_layer.weight, p=2, dim=-1)
            outputs = torch.matmul(outputs_norm, weights_norm.T) / self.temp
        else:
            outputs = torch.matmul(outputs, self.embed_layer.weight.T) / self.temp

        return outputs