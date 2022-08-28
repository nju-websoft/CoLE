import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForMaskedLM, get_cosine_schedule_with_warmup
from torch.optim.adamw import AdamW

from .utils import get_ranks, get_norms, get_scores


class NBert(nn.Module):
    def __init__(self, args: dict, tokenizer: BertTokenizer, bert: BertForMaskedLM):
        super(NBert, self).__init__()

        # 1. hyper params
        self.device = torch.device(args['device'])
        self.pretraining = True if args['task'] == 'pretrain' else False
        self.add_neighbors = True if args['add_neighbors'] else False
        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.lr = args['lm_lr']
        self.entity_begin_idx = args['text_entity_begin_idx']
        self.entity_end_idx = args['text_entity_end_idx']

        # 2. model
        self.tokenizer = tokenizer
        self.bert_encoder = bert
        self.bert_encoder.resize_token_embeddings(len(self.tokenizer))

        self.loss_fc = nn.CrossEntropyLoss(label_smoothing=args['lm_label_smoothing'])

    def forward(self, batch_data):
        if self.pretraining:
            output = self.pretrain(batch_data)
        else:
            output = self.link_prediction(batch_data)
        return output['loss'], output['rank']

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        return loss

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 2)

    def validation_step(self, batch, batch_idx):
        loss, rank = self.forward(batch)
        return loss.item(), rank

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = get_scores(rank, loss)
        return scores

    def pretrain(self, batch):
        # 1. prepare data
        text_data = batch['text_data']
        input_ids = text_data['input_ids'].to(self.device)
        token_type_ids = text_data['token_type_ids'].to(self.device)
        attention_mask = text_data['attention_mask'].to(self.device)
        mask_pos = text_data['mask_pos'].to(self.device)
        labels = batch['labels'].to(self.device)
        filters = None

        # 2. encode with bert
        output = self.bert_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # 3. get logits, calculate loss and rank
        logits = output.logits[mask_pos[:, 0], mask_pos[:, 1], self.entity_begin_idx:self.entity_end_idx]
        loss = self.loss_fc(logits, labels)
        rank = get_ranks(F.softmax(logits, dim=-1), labels, filters)

        return {'loss': loss, 'rank': rank, 'logits': logits}

    def encode_neighbors(self, batch_data):
        # 1. prepare data
        data = batch_data['text_neighbors']
        labels = batch_data['neighbors_labels'].to(self.device)  # (batch_size, )

        # 2. encode neighbors
        loss_from_neighbors = None
        embeds_from_neighbors = []
        for i in range(self.neighbor_num):
            input_ids = data[i]['input_ids'].to(self.device)  # (batch_size, seq_len)
            token_type_ids = data[i]['token_type_ids'].to(self.device)  # (batch_size, seq_len)
            attention_mask = data[i]['attention_mask'].to(self.device)  # (batch_size, seq_len)
            mask_pos = data[i]['mask_pos'].to(self.device)  # (batch_size, 2)
            output = self.bert_encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # 2.1 compute the loss based on neighboring triples
            logits = output.logits[mask_pos[:, 0], mask_pos[:, 1], self.entity_begin_idx: self.entity_end_idx]
            loss = self.loss_fc(logits, labels)
            if loss_from_neighbors is None:
                loss_from_neighbors = loss
            else:
                loss_from_neighbors += loss
            # 2.2 get embeddings of mask_token(batch_size, 768)
            mask_embeds = output.hidden_states[-1][mask_pos[:, 0], mask_pos[:, 1], :]
            embeds_from_neighbors.append(mask_embeds)
        loss = loss_from_neighbors / self.neighbor_num
        embeds = torch.stack(embeds_from_neighbors, dim=0)  # (neighbor_num, batch_size, 768)
        embeds = torch.mean(embeds, dim=0)  # (batch_size, 768)

        return loss, embeds

    def link_prediction(self, batch):
        # 1. prepare data
        text_data = batch['text_data']
        input_ids = text_data['input_ids'].to(self.device)
        token_type_ids = text_data['token_type_ids'].to(self.device)
        attention_mask = text_data['attention_mask'].to(self.device)
        mask_pos = text_data['mask_pos'].to(self.device)
        labels = batch['labels'].to(self.device)
        filters = batch['filters'].to(self.device)

        # 2. aggregate with neighbors
        # input_embeds: (batch_size, seq_len, hidden_size)
        inputs_embeds = self.bert_encoder.bert.embeddings.word_embeddings(input_ids)
        if self.add_neighbors:
            # 2.2 neighbors embeds
            neighbor_loss, neighbor_embeds = self.encode_neighbors(batch)
            unused_token_id = self.tokenizer.convert_tokens_to_ids(self.neighbor_token)
            head_pos = torch.nonzero(torch.eq(input_ids, unused_token_id))
            inputs_embeds[head_pos[:, 0], head_pos[:, 1], :] = neighbor_embeds

        # 3. encode with bert
        output = self.bert_encoder(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # 3. get logits, calculate loss and rank
        logits = output.logits[mask_pos[:, 0], mask_pos[:, 1], self.entity_begin_idx:self.entity_end_idx]
        loss = self.loss_fc(logits, labels)
        rank = get_ranks(F.softmax(logits, dim=-1), labels, filters)

        return {'loss': loss, 'rank': rank, 'logits': logits}

    def configure_optimizers(self, total_steps: int):
        parameters = self.get_parameters()
        opt = AdamW(parameters, eps=1e-6)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        return {'optimizer': opt, 'scheduler': scheduler}

    def get_parameters(self):
        # freeze all layers except word_embeddings when pre-training
        if self.pretraining:
            for n, p in self.bert_encoder.named_parameters():
                if 'word_embeddings' not in n:
                    p.requires_grad = False

        decay_param = []
        no_decay_param = []
        for n, p in self.bert_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if ('bias' in n) or ('LayerNorm.weight' in n):
                no_decay_param.append(p)
            else:
                decay_param.append(p)
        if self.pretraining:  # no_decay_param is empty
            return [
                {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr}
            ]
        else:
            return [
                {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr},
                {'params': no_decay_param, 'weight_decay': 0, 'lr': self.lr}
            ]

    def freeze(self):
        for n, p in self.bert_encoder.named_parameters():
            p.requires_grad = False

    # 保存模型
    def save_model(self, save_dir):
        save_path = os.path.join(save_dir, 'nbert')
        self.bert_encoder.save_pretrained(save_path)

    def clip_grad_norm(self):
        # this function is useless, because gard norm is less that 1.0
        # raise ValueError('Do not call clip_gram_norm for N-BERT')
        if self.pretraining:
            # print('[WARNING] We do not apply clip_grad_norm for pretrain')
            return
        norms = get_norms(self.bert_encoder.parameters()).item()
        info = f'grads for N-BERT: {round(norms, 4)}'
        clip_grad_norm_(self.bert_encoder.parameters(), max_norm=3.0)

        return info

    def grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        return round(norms, 4)



