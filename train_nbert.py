import os
import json
import torch
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time, strftime, localtime
from transformers import BertTokenizer, BertForMaskedLM

from code import KGCDataModule
from code import NBert
from utils import score2str


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def get_args():
    parser = argparse.ArgumentParser()
    # 1. about training
    parser.add_argument('--task', type=str, default='train', help='pretrain | train | validate')
    parser.add_argument('--model_path', type=str, default='checkpoints/fb15k-237/bert-pretrained')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:3', help='select a gpu like cuda:0')
    parser.add_argument('--dataset', type=str, default='fb15k-237', help='select a dataset: fb15k-237 or wn18rr')
    parser.add_argument('--max_seq_length', type=int, default=64, help='max sequence length for inputs to bert')
    # about neighbors
    parser.add_argument('--add_neighbors', action='store_true', default=False)
    parser.add_argument('--neighbor_num', type=int, default=3)
    parser.add_argument('--neighbor_token', type=str, default='[Neighbor]')
    parser.add_argument('--no_relation_token', type=str, default='[R_None]')
    # about text encoder
    parser.add_argument('--lm_lr', type=float, default=5e-5, help='learning rate for language model')
    parser.add_argument('--lm_label_smoothing', type=float, default=0.8, help='label smoothing for language model')
    # 2. some unimportant parameters, only need to change when your server/pc changes, I do not change these
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')

    # 3. convert to dict
    args = parser.parse_args()
    args = vars(args)

    # add some paths: tokenzier_path model_path data_path output_path
    root_path = os.path.dirname(__file__)
    # 1. tokenizer path
    args['tokenizer_path'] = os.path.join(root_path, 'checkpoints', 'bert-base-cased')
    # 2. model path
    args['model_path'] = os.path.join(root_path, args['model_path'])
    # 3. data path
    args['data_path'] = os.path.join(root_path, 'dataset', args['dataset'])  # 数据集目录
    # 4. output path
    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join(root_path, 'output', args['dataset'], 'N-BERT', timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['output_path'] = output_dir

    # save hyper params
    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)

    # set random seed
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    return args


class NBertTrainer:
    def __init__(self, config: dict):
        self.is_validate = True if config['task'] == 'validate' else False
        self.output_path = config['output_path']
        self.epoch = config['epoch']

        tokenizer, self.train_dl, self.dev_dl, self.test_dl = self._load_dataset(config)
        self.model = self._load_model(config, tokenizer).to(config['device'])
        optimizers = self.model.configure_optimizers(total_steps=len(self.train_dl)*self.epoch)
        self.opt, self.scheduler = optimizers['optimizer'], optimizers['scheduler']

        self.log_path = os.path.join(self.output_path, 'log.txt')
        with open(self.log_path, 'w') as f:
            pass

    def _load_dataset(self, config: dict):
        # 1. load tokenizer
        tokenizer_path = config['tokenizer_path']
        print(f'Loading Tokenizer from {tokenizer_path}')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)

        # 2. resize tokenizer, load datasets
        data_module = KGCDataModule(config, tokenizer, encode_text=True)
        tokenizer = data_module.get_tokenizer()
        train_dl = data_module.get_train_dataloader()
        dev_dl = data_module.get_dev_dataloader()
        test_dl = data_module.get_test_dataloader()

        return tokenizer, train_dl, dev_dl, test_dl

    def _load_model(self, config: dict, tokenizer: BertTokenizer):
        text_encoder_path = config['model_path']
        print(f'Loading N-Bert from {text_encoder_path}')
        bert_encoder = BertForMaskedLM.from_pretrained(text_encoder_path)
        model = NBert(config, tokenizer, bert_encoder)
        return model

    def _train_one_epoch(self):
        self.model.train()
        outputs = list()
        for batch_idx, batch_data in enumerate(tqdm(self.train_dl)):
            # 1. forward
            batch_loss = self.model.training_step(batch_data, batch_idx)
            outputs.append(batch_loss)
            # 2. backward
            self.opt.zero_grad()
            batch_loss.backward()
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

    def train(self):
        best_score = None
        for i in range(1, self.epoch + 1):
            begin_time = time()
            train_loss = self._train_one_epoch()
            dev_score = self._validate_one_epoch(self.dev_dl)
            test_score = self._validate_one_epoch(self.test_dl)

            # update the best scores and save the best model
            if best_score is None or best_score['MRR'] < test_score['MRR']:
                best_score = test_score
                best_score['epoch'] = i
                self.model.save_model(self.output_path)

            # save log of this epoch
            log = f'[train] epoch: {i}, loss: {train_loss}' + '\n'
            log += f'[dev]   epoch: {i}, ' + score2str(dev_score) + '\n'
            log += f'[test]  epoch: {i}, ' + score2str(test_score) + '\n'
            log += '=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30 + '\n'
            print(log)
            with open(self.log_path, 'a') as f:
                f.write(log + '\n')
        # save the log of best epoch
        log = f'[best]  epoch: {best_score["epoch"]}, ' + score2str(best_score)
        print(log)
        with open(self.log_path, 'a') as f:
            f.write(log + '\n')

    def validate(self):
        # we do not save log for validation
        shutil.rmtree(self.output_path)

        dev_scores = self._validate_one_epoch(self.dev_dl)
        test_scores = self._validate_one_epoch(self.test_dl)
        print(f'[dev] {score2str(dev_scores)}')
        print(f'[test] {score2str(test_scores)}')

    def main(self):
        if self.is_validate:
            self.validate()
        else:
            self.train()


if __name__ == '__main__':
    config = get_args()
    trainer = NBertTrainer(config)
    trainer.main()

