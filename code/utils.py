import os
import torch
import shutil
import time
import json
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, get_cosine_schedule_with_warmup
from torch.optim.adamw import AdamW

from .data_process.dataset import KGCDataModule
from .models import NBert, NFormer, CoLE, Knowformer


def get_encoder_config(args, vocab_size=-1, **kwargs):
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=args['hidden_size'],
        num_hidden_layers=args['num_hidden_layers'],
        num_attention_heads=args['num_attention_heads'],
        intermediate_size=args['intermediate_size'],
        hidden_act=args['hidden_act'],
        input_dropout_prob=args['input_dropout_prob'],
        context_dropout_prob=args['context_dropout_prob'],
        hidden_dropout_prob=args['hidden_dropout_prob'],
        attention_probs_dropout_prob=args['attention_probs_dropout_prob'],
        max_position_embeddings=args['max_position_embeddings'],
        type_vocab_size=args['type_vocab_size'],
        initializer_range=args['initializer_range'],
        layer_norm_eps=args['layer_norm_eps'],
        pad_token_id=args['pad_token_id'],
        position_embedding_type=args['position_embedding_type'],
        use_cache=args['use_cache'],
        classifier_dropout=args['classifier_dropout'],

        attention_dropout_prob=args['attention_dropout_prob'],
        residual_dropout_prob=args['residual_dropout_prob'],
        num_relations=args['relation_end_idx'] - args['relation_begin_idx'],
        **kwargs,
    )


def get_datasets_and_model(args):
    task = args['task']

    # 1. 加载tokenizer
    tokenizer_path = args['tokenizer_path']
    print(f'Loading Tokenizer from {tokenizer_path}')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_basic_tokenize=False)
    # 2. 加载数据集, 更新tokenizer
    data_module = KGCDataModule(args, tokenizer)
    tokenizer = data_module.get_tokenizer()
    train_dl = data_module.get_train_dataloader()
    dev_dl = data_module.get_dev_dataloader()
    test_dl = data_module.get_test_dataloader()
    args['vocab_size'] = len(tokenizer)
    args['one_epoch_steps'] = len(train_dl)

    # 3. 加载BERT作为 LM_encoder
    if task in ['pretrain', 'mlm']:
        text_encoder_path = args['text_encoder_path']
        print(f'Loading NBert from {text_encoder_path}')
        bert_encoder = BertForMaskedLM.from_pretrained(text_encoder_path)
        model = NBert(args, tokenizer, bert_encoder)
    elif task == 'kge':
        if 'struc_encoder_path' in args:
            struc_encoder_path = args['struc_encoder_path']
            print(f'Loading NFormer from {struc_encoder_path}')
            bert_encoder = Knowformer.from_pretrained(struc_encoder_path)
        else:
            bert_config = get_encoder_config(args, vocab_size=len(tokenizer))
            bert_encoder = Knowformer(bert_config)
        model = NFormer(args, bert_encoder)
    elif task == 'both':
        text_encoder_path = args['text_encoder_path']
        print(f'Loading NBert from {text_encoder_path}')
        bert_encoder = BertForMaskedLM.from_pretrained(text_encoder_path)
        nbert = NBert(args, tokenizer, bert_encoder)

        struc_encoder_path = args['struc_encoder_path']
        print(f'Loading NFormer from {struc_encoder_path}')
        bert_encoder = Knowformer.from_pretrained(struc_encoder_path)
        nformer = NFormer(args, bert_encoder)

        model = CoLE(args, nbert, nformer)
    else:
        assert 0, task

    return {'train': train_dl, 'dev': dev_dl, 'test': test_dl}, model


class Logger:
    def __init__(self, output_dir):
        # 1. 创建根目录
        # if not os.path.exists(output_dir):  # 对应数据集  root_path/output/fb15k-237/
        #     os.makedirs(output_dir)
        # self.timestamp = time.strftime('%m-%d-%H-%M', time.localtime())
        # self.output_dir = os.path.join(output_dir, self.timestamp)
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
        # os.makedirs(self.output_dir)
        self.output_dir = output_dir

        # 保存输出日志
        self.log_path = os.path.join(self.output_dir, 'log.txt')
        with open(self.log_path, 'w') as f:
            pass

    def save_args(self, args):
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(args, f, indent=4)

    def print(self, string):
        print(string)
        with open(self.log_path, 'a') as f:
            f.write(string + '\n')

