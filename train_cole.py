import os
from time import time, strftime, localtime
from tqdm import tqdm
import json
import torch
import shutil
import random
import argparse
import numpy as np

from transformers import BertTokenizer, BertForMaskedLM, BertConfig

from code import KGCDataModule
from code import NBert, NFormer, CoLE
from utils import save_model, load_model, score2str


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


MODEL_PATH = {
    'fb15k-237': {
        'nbert': 'output/fb15k-237/N-BERT/20220824_132950/nbert',
        'nformer': 'output/fb15k-237/N-Former/20220823_202921/avg.bin',
        # 'nbert': 'output/fb15k-237/CoLE/20220827_222516/nbert',
        # 'nformer': 'output/fb15k-237/CoLE/20220827_222516/nformer.bin',
    },
    'wn18rr': {
        'nbert': 'checkpoints/wn18rr/',
        'nformer': 'output/wn18rr/',
    }
}


def get_args():
    parser = argparse.ArgumentParser()
    # 1. about training
    parser.add_argument('--task', type=str, default='train', help='train | validate')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:3', help='select a gpu like cuda:0')
    parser.add_argument('--dataset', type=str, default='fb15k-237', help='select a dataset: fb15k-237 or wn18rr')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--max_seq_length', type=int, default=64, help='max sequence length for inputs to bert')
    # about neighbors
    parser.add_argument('--extra_encoder', action='store_true', default=False)
    parser.add_argument('--add_neighbors', action='store_true', default=False)
    parser.add_argument('--neighbor_num', type=int, default=3)
    parser.add_argument('--neighbor_token', type=str, default='[Neighbor]')
    parser.add_argument('--no_relation_token', type=str, default='[R_None]')
    # text encoder
    parser.add_argument('--lm_lr', type=float, default=1e-5, help='learning rate for language model')
    parser.add_argument('--lm_label_smoothing', type=float, default=0.8, help='label smoothing for language model')
    parser.add_argument('--lm_max_grad_norm', type=float, default=3.0, help='max norm for language model')
    # struc encoder
    parser.add_argument('--kge_lr', type=float, default=5e-5)
    parser.add_argument('--kge_label_smoothing', type=float, default=0.8)
    parser.add_argument('--kge_max_grad_norm', type=float, default=1.0)
    # struc encoder config
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--input_dropout_prob', type=float, default=0.7)
    parser.add_argument('--attention_dropout_prob', type=float, default=0.1)
    parser.add_argument('--context_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--residual_dropout_prob', type=float, default=0.)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    # 2. unimportant parameters, only need to change when GPUs change
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    # 3. convert to dict
    args = parser.parse_args()
    args = vars(args)

    # add some paths: tokenzier text_encoder data_dir output_dir
    root_path = os.path.dirname(__file__)
    # 1. tokenizer path
    args['tokenizer_path'] = os.path.join(root_path, 'checkpoints', 'bert-base-cased')
    # 2. model path
    args['nbert_path'] = os.path.join(root_path, MODEL_PATH[args['dataset']]['nbert'])
    args['nformer_path'] = os.path.join(root_path, MODEL_PATH[args['dataset']]['nformer'])
    # 3. data path
    args['data_path'] = os.path.join(root_path, 'dataset', args['dataset'])  # 数据集目录
    # 4. output path
    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join(root_path, 'output', args['dataset'], 'CoLE', timestamp)
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

    return args


class CoLETrainer:
    def __init__(self, config: dict):
        self.is_validate = True if config['task'] == 'validate' else False
        self.output_path = config['output_path']
        self.epoch = config['epoch']

        tokenizer, self.train_dl, self.dev_dl, self.test_dl = self._load_dataset(config)
        self.model_config, self.model = self._load_model(config, tokenizer)
        opt1, opt2 = self.model.configure_optimizers(total_steps=len(self.train_dl)*self.epoch)
        self.text_opt, self.text_sche = opt1['optimizer'], opt1['scheduler']
        self.struc_opt, self.struc_sche = opt2['optimizer'], opt2['scheduler']

        self.log_path = os.path.join(self.output_path, 'log.txt')
        with open(self.log_path, 'w') as f:
            pass

    def _load_dataset(self, config: dict):
        # 1. load tokenizer
        tokenizer_path = config['tokenizer_path']
        print(f'Loading Tokenizer from {tokenizer_path}')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)

        # 2. resize tokenizer, load datasets
        data_module = KGCDataModule(config, tokenizer, encode_text=True, encode_struc=True)
        tokenizer = data_module.get_tokenizer()
        train_dl = data_module.get_train_dataloader()
        dev_dl = data_module.get_dev_dataloader()
        test_dl = data_module.get_test_dataloader()

        return tokenizer, train_dl, dev_dl, test_dl

    def _load_model(self, config: dict, tokenizer: BertTokenizer):
        # 1. load N-BERT
        text_encoder_path = config['nbert_path']
        print(f'Loading N-Bert from {text_encoder_path}')
        bert_encoder = BertForMaskedLM.from_pretrained(text_encoder_path)
        nbert = NBert(config, tokenizer, bert_encoder)
        # 2. load N-Former
        nformer_path = config["nformer_path"]
        model_config, bert_encoder = load_model(nformer_path, config['device'])
        nformer = NFormer(config, bert_encoder).to(config['device'])

        model = CoLE(config, nbert, nformer).to(config['device'])
        return model_config, model

    def _train_one_epoch(self):
        self.model.train()
        outputs = list()
        for batch_idx, batch_data in enumerate(tqdm(self.train_dl)):
            batch_loss = self.model.training_step(batch_data, batch_idx)
            outputs.append(batch_loss)
            batch_text_loss, batch_struc_loss = batch_loss

            self.text_opt.zero_grad()
            batch_text_loss.backward()
            self.text_opt.step()
            if self.text_sche is not None:
                self.text_sche.step()
            self.text_opt.zero_grad()

            self.struc_opt.zero_grad()
            batch_struc_loss.backward()
            self.struc_opt.step()
            if self.struc_sche is not None:
                self.struc_sche.step()
            self.struc_opt.zero_grad()
        text_loss, struc_loss = self.model.training_epoch_end(outputs)
        return text_loss, struc_loss

    def _validate_one_epoch(self, data_loader):
        self.model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(data_loader)):
                output = self.model.validation_step(batch_data, batch_idx)
                outputs.append(output)

        text_score, struc_score, cole_score = self.model.validation_epoch_end(outputs)
        return text_score, struc_score, cole_score

    def train(self):
        text_score, struc_score, cole_score = self._validate_one_epoch(self.test_dl)
        log = f'[NBert]   {score2str(text_score)}\n'
        log += f'[NFormer] {score2str(struc_score)}\n'
        log += f'[CoLE]    {score2str(cole_score)}\n'
        log += '=' * 30 + 'validation before train' + '=' * 30 + '\n'
        print(log)
        with open(self.log_path, 'a') as f:
            f.write(log + '\n')

        text_best_mrr = 0.
        struc_best_mrr = 0.
        for i in range(1, self.epoch+1):
            begin_time = time()
            text_loss, struc_loss = self._train_one_epoch()
            text_score, struc_score, cole_score = self._validate_one_epoch(self.test_dl)

            # save NBert
            if text_score['MRR'] > text_best_mrr:
                text_best_mrr = text_score['MRR']
                self.model.text_encoder.save_model(self.output_path)
            # save NFormer
            if struc_score['MRR'] > struc_best_mrr:
                struc_best_mrr = struc_score['MRR']
                save_model(self.model_config, self.model.struc_encoder.bert_encoder,
                           os.path.join(self.output_path, 'nformer.bin'))

            # write logs
            log = f'[train]    epoch: {i}, text_loss: {text_loss}, struc_loss: {struc_loss}\n'
            log += f'[N-BERT]   epoch: {i}, ' + score2str(text_score) + '\n'
            log += f'[N-Former] epoch: {i}, ' + score2str(struc_score) + '\n'
            log += '=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30 + '\n'
            print(log)
            with open(self.log_path, 'a') as f:
                f.write(log + '\n')

    def validate(self):
        # we do not save log for validation
        shutil.rmtree(self.output_path)

        text_score, struc_score, cole_score = self._validate_one_epoch(self.dev_dl)
        log = f'[NBert]   {score2str(text_score)}\n'
        log += f'[NFormer] {score2str(struc_score)}\n'
        log += f'[CoLE]    {score2str(cole_score)}\n'
        print(log)

        text_score, struc_score, cole_score = self._validate_one_epoch(self.test_dl)
        log = f'[NBert]   {score2str(text_score)}\n'
        log += f'[NFormer] {score2str(struc_score)}\n'
        log += f'[CoLE]    {score2str(cole_score)}\n'
        print(log)

    def main(self):
        if self.is_validate:
            self.validate()
        else:
            self.train()


if __name__ == '__main__':
    config = get_args()
    trainer = CoLETrainer(config)
    trainer.main()

