import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup
from contiguous_params import ContiguousParams

from .knowformer import Knowformer
from .utils import get_ranks, get_norms, get_scores


class NFormer(nn.Module):
    def __init__(self, args: dict, bert_encoder: Knowformer):
        super(NFormer, self).__init__()

        # 1. 保存有用的数据
        self.device = torch.device(args['device'])
        self.add_neighbors = True if args['add_neighbors'] else False
        self.neighbor_num = args['neighbor_num']
        self.lr = args['kge_lr']
        self.entity_begin_idx = args['struc_entity_begin_idx']
        self.entity_end_idx = args['struc_entity_end_idx']
        self.use_extra_encoder = args['extra_encoder']

        self.bert_encoder = bert_encoder

        self.loss_fc = nn.CrossEntropyLoss(label_smoothing=args['kge_label_smoothing'])

    def forward(self, batch_data):
        output = self.link_prediction(batch_data)
        return output['loss'], output['rank']

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        return loss

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 4)

    def validation_step(self, batch, batch_idx):
        output = self.link_prediction_validation(batch)
        loss, rank = output['loss'], output['rank']
        return loss.item(), rank

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = get_scores(rank, loss)
        return scores

    def link_prediction_validation(self, batch):
        # validate without neighboring triplets will get a higher score

        # 1. prepare data
        input_ids = batch['struc_data']['input_ids'].to(self.device)  # batch_size * 3
        context_input_ids = None
        labels = batch['labels'].to(self.device)  # batch_size
        filters = batch['filters'].to(self.device)

        # 2. get output from knowformer
        output = self.bert_encoder(input_ids, context_input_ids, self.use_extra_encoder)
        origin_logits = output['without_neighbors']

        # 3. compute loss and rank
        origin_loss = self.loss_fc(origin_logits, labels + self.entity_begin_idx)
        origin_logits = origin_logits[:, self.entity_begin_idx: self.entity_end_idx]
        rank = get_ranks(F.softmax(origin_logits, dim=-1), labels, filters)
        return {'loss': origin_loss, 'rank': rank, 'logits': origin_logits}

    def link_prediction(self, batch):
        # 1. prepare data
        input_ids = batch['struc_data']['input_ids'].to(self.device)  # batch_size * 3
        if self.add_neighbors:
            context_input_ids = [t['input_ids'].to(self.device) for t in batch['struc_neighbors']]
            neighbors_labels = batch['neighbors_labels'].to(self.device)
        else:
            context_input_ids = None
            neighbors_labels = None
        labels = batch['labels'].to(self.device)  # batch_size
        filters = batch['filters'].to(self.device)

        # 2. encode
        output = self.bert_encoder(input_ids, context_input_ids, self.use_extra_encoder)
        origin_logits = output['without_neighbors']
        mixed_logits = output['with_neighbors']
        context_logits = output['neighbors']

        # 3. compute lossed
        # 3.1 loss from the current triplet
        origin_loss = self.loss_fc(origin_logits, labels + self.entity_begin_idx)
        if not self.add_neighbors:
            origin_logits = origin_logits[:, self.entity_begin_idx: self.entity_end_idx]
            rank = get_ranks(F.softmax(origin_logits, dim=-1), labels, filters)
            return {'loss': origin_loss, 'rank': rank, 'logits': origin_logits}

        # 3.2 losses from neighboring triplets
        loss_for_neighbors = None
        for i in range(self.neighbor_num):
            logits = context_logits[i]
            loss = self.loss_fc(logits, neighbors_labels + self.entity_begin_idx)
            if loss_for_neighbors is None:
                loss_for_neighbors = loss
            else:
                loss_for_neighbors += loss
        loss_for_neighbors = loss_for_neighbors / self.neighbor_num
        # 3.3 loss from mixed embeddings
        mixed_loss = self.loss_fc(mixed_logits, labels + self.entity_begin_idx)
        # 3.4 merge all losses
        loss = origin_loss + mixed_loss + 0.5 * loss_for_neighbors

        logits = mixed_logits[:, self.entity_begin_idx: self.entity_end_idx] \
                 + origin_logits[:, self.entity_begin_idx: self.entity_end_idx]
        rank = get_ranks(F.softmax(logits, dim=-1), labels, filters)

        return {'loss': loss, 'rank': rank, 'logits': logits}

    ####################################################################################################################
    def configure_optimizers(self, total_steps: int):
        opt = torch.optim.AdamW(ContiguousParams(self.bert_encoder.parameters()).contiguous(), lr=self.lr)
        return {'optimizer': opt, 'scheduler': None}

    def get_parameters(self):
        decay_param = []
        no_decay_param = []
        for n, p in self.bert_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if ('bias' in n) or ('LayerNorm.weight' in n):
                no_decay_param.append(p)
            else:
                decay_param.append(p)
        return [
            {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr},
            {'params': no_decay_param, 'weight_decay': 0, 'lr': self.lr}
        ]

    def freeze(self):
        for n, p in self.bert_encoder.named_parameters():
            p.requires_grad = False

    def clip_grad_norm(self):
        # this function is useless, because gard norm is less that 1.0
        # raise ValueError('Do not call clip_gram_norm for N-Former')
        norms = get_norms(self.bert_encoder.parameters()).item()
        info = f'grads for N-Former: {round(norms, 4)}'
        # clip_grad_norm_(self.bert_encoder.parameters(), max_norm=self.max_norm)
        return info

    def grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        return round(norms, 4)

