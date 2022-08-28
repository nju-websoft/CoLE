import json
import os
import shutil
from transformers import BertTokenizer

"""
该文件暂时废弃
简化数据集, 原始数据集包含很多无用的信息, 此处将其省略, 只保留想要的信息
简化后数据集的格式为(sub, obj, prompt, ...)
"""


def convert_GoogleRE(src_path: str, dest_path: str):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)

    for file in os.listdir(src_path):
        new_dataset = list()
        with open(os.path.join(src_path, file), 'r') as f:
            for line in f:
                data_dit = json.loads(line)
                if file.startswith('date_of_birth'):
                    prompt = '[X] (born [Y]).'
                elif file.startswith('place_of_birth'):
                    prompt = '[X] was born in [Y] .'
                elif file.startswith('place_of_death'):
                    prompt = '[X] died in [Y] .'
                else:
                    assert 0, 'Unknown relation'
                new_dataset.append({'sub': data_dit['sub_label'],
                                    'obj': data_dit['obj_label'],
                                    'prompt': prompt})

        with open(os.path.join(dest_path, file), 'w') as f:
            for data_dit in new_dataset:
                json.dump(data_dit, f)
                f.write('\n')


def convert_FB15k_237(src_path: str, dest_path: str):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)

    tokenizer = BertTokenizer.from_pretrained(model_path)
    for file in os.listdir(src_path):
        single_dataset = list()
        multi_dataset = list()
        with open(os.path.join(src_path, file), 'r') as f:
            for line in f:
                sub, relation, obj = line.strip().split('\t')
                prompt = f"[X] {relation.split('/')[-1].replace('_', ' ')} [Y]."
                data_dit = {'sub': sub, 'obj': obj, 'prompt': prompt}
                if len(tokenizer.tokenize(obj)) == 1:
                    single_dataset.append(data_dit)
                else:
                    multi_dataset.append(data_dit)

        with open(os.path.join(dest_path, file.replace('.txt', '') + 'single.jsonl'), 'w') as f:
            for data_dit in single_dataset:
                json.dump(data_dit, f)
                f.write('\n')
        with open(os.path.join(dest_path, file.replace('.txt', '') + 'multi.jsonl'), 'w') as f:
            for data_dit in multi_dataset:
                json.dump(data_dit, f)
                f.write('\n')


def convert_mlama(src_path: str, dest_path: str):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)

    relation_prompt = dict()
    with open(os.path.join(src_path, 'templates.jsonl'), 'r') as f:
        for line in f:
            data_dit = json.loads(line)
            relation_prompt[data_dit['relation']] = data_dit['template']

    new_dataset = list()
    for file in os.listdir(src_path):
        if file.startswith('templates'):
            continue
        relation = file.split('.')[0]
        with open(os.path.join(src_path, file), 'r') as f:
            for line in f:
                data_dit = json.loads(line)
                prompt = relation_prompt[relation]
                new_dataset.append({'sub': data_dit['sub_label'],
                                    'obj': data_dit['obj_label'],
                                    'prompt': prompt})

    with open(os.path.join(dest_path, 'mlama.jsonl'), 'w') as f:
        for data_dit in new_dataset:
            json.dump(data_dit, f)
            f.write('\n')


def convert_WN18RR(src_path: str, dest_path: str):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)

    entity = set()
    for file in ['train.txt', 'valid.txt', 'test.txt']:
        with open(os.path.join(src_path, file), 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                sub, obj = tokens[0], tokens[-1]
                entity.add(sub)
                entity.add(obj)
    print(f'实体数量： {len(entity)}')

    wordnet_dir = os.path.join(root_path, 'dataset/source_dataset/WordNet')
    id2noun = read_wordnet(os.path.join(wordnet_dir, 'data.noun'))
    id2verb = read_wordnet(os.path.join(wordnet_dir, 'data.verb'))
    id2adj = read_wordnet(os.path.join(wordnet_dir, 'data.adj'))
    id2adv = read_wordnet(os.path.join(wordnet_dir, 'data.adv'))
    print(f'名词数量: {len(id2noun)}')
    print(f'动词数量: {len(id2verb)}')
    print(f'形容词数量: {len(id2adj)}')
    print(f'副词数量: {len(id2adv)}')

    cnt = [0, 0, 0, 0, 0]
    for entity_id in entity:
        if entity_id in id2noun:
            cnt[0] += 1
        elif entity_id in id2verb:
            cnt[1] += 1
        elif entity_id in id2adj:
            cnt[2] += 1
        elif entity_id in id2adv:
            cnt[3] += 1
        else:
            cnt[4] += 1
    print(cnt)


def read_wordnet(file):
    id2word = dict()
    with open(file, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            id = tokens[0]
            word_num = int(tokens[3], 16)
            words = []
            for i in range(4, 4 + 2 * word_num, 2):
                words.append(tokens[i])
            desc = line.split('|')[-1].strip().split(';')[0]
            id2word[id] = {'words': words, 'description': desc}
    return id2word


if __name__ == '__main__':
    root_path = os.path.join(os.getcwd(), '../..')
    model_path = os.path.join(os.getcwd(), '../../pretrained_models/bert_base_cased')
    bert_tokenizer = BertTokenizer.from_pretrained(model_path)

    # src_path = os.path.join(root_path, 'dataset/source_dataset/WN18RR')
    # dest_path = os.path.join(root_path, 'dataset/WN18RR')
    # convert_WN18RR(src_path, dest_path)

    # src_path = os.path.join(root_path, 'dataset/source_dataset/CoDEx')
    # dest_path = os.path.join(root_path)
    # convert_CoDEx(src_path, dest_path)

    # src_path = os.path.join(root_path, 'dataset/source_dataset/mlama1.1')
    # dest_path = os.path.join(root_path, 'dataset/mlama')
    # convert_mlama(src_path, dest_path)

    # src_path = os.path.join(root_path, 'dataset/source_dataset/FB15k_237')
    # dest_path = os.path.join(root_path, 'dataset/FB15k_237')
    # convert_FB15k_237(src_path, dest_path)

    # src_path = os.path.join(root_path, 'dataset/source_dataset/Google_RE')
    # dest_path = os.path.join(root_path, 'dataset/Google_RE')
    # convert_GoogleRE(src_path, dest_path)
