import os
import json
import torch
import shutil
import random
import argparse
import numpy as np
from time import time, strftime, localtime
from transformers import BertTokenizer

from code import KGCDataModule
from code import NFormer, Knowformer
from utils import save_model, load_model, score2str, swa


'''
train:
run 'python train_nformer.py --add_neighbors' to train N-Former for link prediction by default hyper-parameters
we save the best 20 models for swa, the best model is named nformer.bin, the swa model is named avg.bin

validate:
copy the trained model to MODEL_PATH
run 'python train_nformer.py --task validate'
'''


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
MODEL_PATH = {
    'fb15k-237': 'output/fb15k-237/N-Former/20220823_202921/avg.bin',
    # 'fb15k-237': 'output/fb15k-237/CoLE/20220827_222516/nformer.bin',
    'wn18rr': 'output/wn18rr/',
}


def get_args():
    parser = argparse.ArgumentParser()
    # 1. about training
    parser.add_argument('--task', type=str, default='train', help='train | validate')
    parser.add_argument('--epoch', type=int, default=600, help='epoch')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:3', help='select a gpu like cuda:0')
    parser.add_argument('--dataset', type=str, default='fb15k-237', help='select a dataset: fb15k-237 or wn18rr')
    # about neighbors
    parser.add_argument('--extra_encoder', action='store_true', default=False)
    parser.add_argument('--add_neighbors', action='store_true', default=False)
    parser.add_argument('--neighbor_num', type=int, default=3)
    parser.add_argument('--neighbor_token', type=str, default='[Neighbor]')
    parser.add_argument('--no_relation_token', type=str, default='[R_None]')
    # about struc encoder
    parser.add_argument('--kge_lr', type=float, default=2e-4)
    parser.add_argument('--kge_label_smoothing', type=float, default=0.8)
    # struc encoder config的参数
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--input_dropout_prob', type=float, default=0.7, help='dropout before encoder')
    parser.add_argument('--context_dropout_prob', type=float, default=0.1, help='dropout for embeddings from neighbors')
    parser.add_argument('--attention_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--residual_dropout_prob', type=float, default=0.)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    # 2. some unimportant parameters, only need to change when your server/pc changes, I do not change these
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    # 3. convert to dict
    args = parser.parse_args()
    args = vars(args)

    # add some paths: tokenzier text_encoder data_dir output_dir
    root_path = os.path.dirname(__file__)
    # 1. tokenizer path
    args['tokenizer_path'] = os.path.join(root_path, 'checkpoints', 'bert-base-cased')
    # 2. saved model_path
    if args['task'] == 'validate':
        args['model_path'] = os.path.join(root_path, MODEL_PATH[args['dataset']])
    # 3. data path
    args['data_path'] = os.path.join(root_path, 'dataset', args['dataset'])
    # 4. output path
    timestamp = strftime('%Y%m%d_%H%M%S', localtime())
    output_dir = os.path.join(root_path, 'output', args['dataset'], 'N-Former', timestamp)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['output_path'] = output_dir

    # save hyper params
    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)
    for k in args:
        print(f'{k}: {args[k]}')

    # set random seed
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return args


def get_model_config(config):
    model_config = dict()
    model_config["hidden_size"] = config['hidden_size']
    model_config["num_hidden_layers"] = config['num_hidden_layers']
    model_config["num_attention_heads"] = config['num_attention_heads']
    model_config["input_dropout_prob"] = config['input_dropout_prob']
    model_config["attention_dropout_prob"] = config['attention_dropout_prob']
    model_config["hidden_dropout_prob"] = config['hidden_dropout_prob']
    model_config["residual_dropout_prob"] = config['residual_dropout_prob']
    model_config["context_dropout_prob"] = config['context_dropout_prob']
    model_config["initializer_range"] = config['initializer_range']
    model_config["intermediate_size"] = config['intermediate_size']

    model_config["vocab_size"] = config['vocab_size']
    model_config["num_relations"] = config['num_relations']
    return model_config


class NFormerTrainer:
    def __init__(self, config: dict):
        self.is_validate = True if config['task'] == 'validate' else False
        self.output_path = config['output_path']
        self.epoch = config['epoch']
        self.device = config['device']

        tokenizer, self.train_dl, self.dev_dl, self.test_dl = self._load_dataset(config)
        self.model_config, self.model = self._load_model(config)

        optimizers = self.model.configure_optimizers(total_steps=self.epoch)
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
        data_module = KGCDataModule(config, tokenizer, encode_struc=True)
        tokenizer = data_module.get_tokenizer()
        train_dl = data_module.get_train_dataloader()
        dev_dl = data_module.get_dev_dataloader()
        test_dl = data_module.get_test_dataloader()

        return tokenizer, train_dl, dev_dl, test_dl

    def _load_model(self, config):
        if self.is_validate:
            model_config, bert_encoder = load_model(config['model_path'], config['device'])
            model = NFormer(config, bert_encoder).to(config['device'])
            return model_config, model
        else:
            model_config = get_model_config(config)
            bert_encoder = Knowformer(model_config)
            model = NFormer(config, bert_encoder).to(config['device'])
            return model_config, model

    def _train_one_epoch(self):
        self.model.train()
        outputs = list()
        for batch_idx, batch_data in enumerate(self.train_dl):
            batch_loss = self.model.training_step(batch_data, batch_idx)
            outputs.append(batch_loss)

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
            for batch_idx, batch_data in enumerate(data_loader):
                output = self.model.validation_step(batch_data, batch_idx)
                outputs.append(output)

        return self.model.validation_epoch_end(outputs)

    def train(self):
        best_score = None
        swa_num = 20
        top_hits1 = [0.] * swa_num
        top_idxs = [-1] * swa_num
        for i in range(1, self.epoch + 1):
            # 1. train and validate one epoch
            begin_time = time()
            train_loss = self._train_one_epoch()
            dev_score = self._validate_one_epoch(self.dev_dl)
            test_score = self._validate_one_epoch(self.test_dl)

            # 2. save log of this epoch
            log = f'[train] epoch: {i}, loss: {train_loss}' + '\n'
            log += f'[dev]   epoch: {i}, ' + score2str(dev_score) + '\n'
            log += f'[test]  epoch: {i}, ' + score2str(test_score) + '\n'
            log += '=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30 + '\n'
            print(log)
            with open(self.log_path, 'a') as f:
                f.write(log + '\n')

            # 3. update the best scores, save the best model
            if best_score is None or best_score['MRR'] < test_score['MRR']:
                best_score = test_score
                best_score['epoch'] = i
                save_model(self.model_config, self.model.bert_encoder, os.path.join(self.output_path, 'nformer.bin'))

            # 4. save the best 20 models for SWA
            min_hits1 = min(top_hits1)
            if test_score['hits@1'] > min_hits1:
                min_idx = top_hits1.index(min_hits1)
                top_hits1[min_idx] = test_score['hits@1']
                old_epoch = top_idxs[min_idx]
                top_idxs[min_idx] = i

                path1 = os.path.join(self.output_path, f'epoch_{old_epoch}.bin')
                path2 = os.path.join(self.output_path, f'epoch_{i}.bin')
                if os.path.exists(path1):
                    os.remove(path1)
                save_model(self.model_config, self.model.bert_encoder, path2)

        # save the log of best epoch, after training
        log = f'[best]  epoch: {best_score["epoch"]}, ' + score2str(best_score) + '\n'
        print(log)
        with open(self.log_path, 'a') as f:
            f.write(log + '\n')

        # save the averaged model by SWA
        swa(self.output_path, self.device)

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
    # swa('output/fb15k-237/N-Former/20220822_153504', 'cpu')
    # assert 0
    config = get_args()
    trainer = NFormerTrainer(config)
    trainer.main()

