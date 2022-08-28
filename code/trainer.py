import json
import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
from time import time

from .utils import get_datasets_and_model, Logger
from .models import NBert, NFormer, CoLE


def convert_scores_to_string(scores):
    loss = scores['loss']
    hits1 = scores['hits@1']
    hits3 = scores['hits@3']
    hits10 = scores['hits@10']
    mrr = scores['MRR']
    return f'loss: {loss}, hits@1: {hits1}, hits@3: {hits3}, hits@10: {hits10}, MRR: {mrr}'


def save_results(triples, ranks):
    results = list()
    batch_size = len(triples)
    for i in range(batch_size):
        h, r, t = triples[i]
        rank = ranks[i]
        results.append((h, r, t, rank))
    return results


class Trainer:
    def __init__(self, args: dict):
        self.args = args
        self.output_dir = args['output_dir']

        # 定义保存分数的logger
        self.logger = Logger(self.output_dir)
        self.logger.save_args(args)

        # 定义数据集和模型, 数据集会影响模型的词表, 因此使用一个函数同时定义数据集和模型
        datasets, model = get_datasets_and_model(args)
        self.train_dl, self.dev_dl, self.test_dl = datasets['train'], datasets['dev'], datasets['test']
        self.model = model.to(torch.device(args['device']))
        # 定义优化器
        optimizers = self.model.configure_optimizers(total_steps=args['one_epoch_steps'])
        self.opt, self.scheduler = optimizers['optimizer'], optimizers['scheduler']

        # self.results = list()

    def _train_one_epoch(self):
        self.model.train()
        outputs = list()
        for batch_idx, batch_data in enumerate(tqdm(self.train_dl)):
            batch_loss = self.model.training_step(batch_data, batch_idx)
            outputs.append(batch_loss)

            self.opt.zero_grad()
            batch_loss.backward()
            self.model.clip_grad_norm()
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
        loss = self.model.training_epoch_end(outputs)
        return loss

    def _validate_one_epoch(self, data_loader):
        self.model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(data_loader)):
                output = self.model.validation_step(batch_data, batch_idx)
                outputs.append(output)

        return self.model.validation_epoch_end(outputs)

    def train_both(self):
        text_scores, struc_scores, cole_scores = self._validate_one_epoch(self.test_dl)
        self.logger.print(f'[NBert]   {convert_scores_to_string(text_scores)}')
        self.logger.print(f'[NFormer] {convert_scores_to_string(struc_scores)}')
        self.logger.print(f'[CoLE]    {convert_scores_to_string(cole_scores)}')
        self.logger.print('=' * 30 + 'validation before train' + '=' * 30)

        text_best_mrr = 0.
        struc_best_mrr = 0.
        for i in range(1, self.args['epoch']+1):
            begin_time = time()
            train_loss = self._train_one_epoch()
            text_scores, struc_scores, cole_scores = self._validate_one_epoch(self.test_dl)
            # save NBert
            if text_scores['MRR'] > text_best_mrr:
                text_best_mrr = text_scores['MRR']
                self.model.text_encoder.save_model(self.output_dir)
            # save NFormer
            if struc_scores['MRR'] > struc_best_mrr:
                struc_best_mrr = struc_scores['MRR']
                self.model.struc_encoder.save_model(self.output_dir)
            # write logs
            self.logger.print(f'[train] epoch: {i}, loss: {train_loss}')
            self.logger.print(f'[mlm]   epoch: {i}, ' + convert_scores_to_string(text_scores))
            self.logger.print(f'[kge]   epoch: {i}, ' + convert_scores_to_string(struc_scores))
            self.logger.print('=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30)

    def train(self):
        best_scores = None
        for i in range(1, self.args['epoch']+1):
            begin_time = time()
            train_loss = self._train_one_epoch()
            dev_scores = self._validate_one_epoch(self.dev_dl)
            test_scores = self._validate_one_epoch(self.test_dl)

            # update the best scores
            if best_scores is None or best_scores['MRR'] < test_scores['MRR']:
                best_scores = test_scores
                best_scores['epoch'] = i
                self.model.save_model(self.logger.output_dir)

            # print the scores of this epoch
            self.logger.print(f'[train] epoch: {i}, loss: {train_loss}')
            self.logger.print(f'[dev]   epoch: {i}, ' + convert_scores_to_string(dev_scores))
            self.logger.print(f'[test]  epoch: {i}, ' + convert_scores_to_string(test_scores))
            self.logger.print(f'[best]  epoch: {best_scores["epoch"]}, ' + convert_scores_to_string(best_scores))
            self.logger.print('=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30)

    def validate(self):
        dev_scores = self._validate_one_epoch(self.dev_dl)
        test_scores = self._validate_one_epoch(self.test_dl)
        if type(dev_scores) is dict:
            print(f'[dev] {convert_scores_to_string(dev_scores)}')
            print(f'[test] {convert_scores_to_string(test_scores)}')
        else:
            text_scores, struc_scores, cole_scores = dev_scores
            print(f'[NBert] [dev]   {convert_scores_to_string(text_scores)}')
            print(f'[NFormer] [dev] {convert_scores_to_string(struc_scores)}')
            print(f'[CoLE] [dev]    {convert_scores_to_string(cole_scores)}')
            text_scores, struc_scores, cole_scores = test_scores
            print(f'[NBert] [test]   {convert_scores_to_string(text_scores)}')
            print(f'[NFormer] [test] {convert_scores_to_string(struc_scores)}')
            print(f'[CoLE] [test]    {convert_scores_to_string(cole_scores)}')

    def main(self):
        if self.args['validation']:
            self.validate()
        else:
            if self.args['task'] == 'both':
                self.train_both()
            else:
                self.train()



