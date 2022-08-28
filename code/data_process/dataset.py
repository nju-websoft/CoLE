import os
import copy
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


# nothing special in this class
class KGCDataset(Dataset):
    def __init__(self, data: list):
        super(KGCDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]


# prepare the dataset
class KGCDataModule:
    def __init__(self, args: dict, tokenizer, encode_text=False, encode_struc=False):
        # 0. some variables used in this class
        self.task = args['task']
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.max_seq_length = args['max_seq_length']

        self.add_neighbors = args['add_neighbors']
        self.neighbor_num = args['neighbor_num']
        self.neighbor_token = args['neighbor_token']
        self.no_relation_token = args['no_relation_token']

        self.encode_text = encode_text
        self.encode_struc = encode_struc

        # 1. read entities and relations from files
        self.entities, self.relations = self.read_support()
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}')

        # 2.1 expand the tokenizer for BERT
        self.tokenizer = tokenizer
        text_offset = self.resize_tokenizer()
        # 2.2 construct the vocab for our KGE module
        self.vocab, struc_offset = self.get_vocab()
        # 2.3 offsets indicate the positions of entities in the vocab, we store them in args and pass to other classes
        args.update(text_offset)
        args.update(struc_offset)
        # 2.4 the following two variables will be used to construct the KGE module
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = struc_offset['struc_relation_end_idx'] - struc_offset['struc_relation_begin_idx']

        # 3.1 read the dataset
        self.lines = self.read_lines()  # {'train': [(h,r,t),...], 'dev': [], 'test': []}
        # 3.2 use the training set to get neighbors
        self.neighbors = self.get_neighbors()  # {ent: {text_prompt: [], struc_prompt: []}, ...}
        # 3.3 entities to be filtered when predict some triplet
        self.entity_filter = self.get_entity_filter()

        # 5. use the triplets in the dataset to construct the inputs for our BERT and KGE module
        if self.task == 'pretrain':
            # utilize entities to get dataset when task is pretrain
            examples = self.create_pretrain_examples()
        else:
            examples = self.create_examples()

        # 6. the above inputs are used to construct pytorch Dataset objects
        self.train_ds = KGCDataset(examples['train'])
        self.dev_ds = KGCDataset(examples['dev'])
        self.test_ds = KGCDataset(examples['test'])

    # the following six functions are called in the __init__ function
    def read_support(self):
        """
        read entities and relations from files
        :return: two Python Dict objects
        """
        entity_path = os.path.join(self.data_path, 'support', 'entity.json')
        entities = json.load(open(entity_path, 'r', encoding='utf-8'))
        for idx, e in enumerate(entities):  # 14541
            new_name = f'[E_{idx}]'
            raw_name = entities[e]['name']
            desc = entities[e]['desc']
            entities[e] = {
                'token_id': idx,  # used for filtering
                'name': new_name,  # new token to be added in tokenizer because raw name may consist many tokens
                'desc': desc,  # entity description, which improve the performance significantly
                'raw_name': raw_name,  # meaningless for the model, but can be used to print texts for debugging
            }

        relation_path = os.path.join(self.data_path, 'support', 'relation.json')
        relations = json.load(open(relation_path, 'r', encoding='utf-8'))
        for idx, r in enumerate(relations):  # 237
            sep1, sep2, sep3, sep4 = f'[R_{idx}_SEP1]', f'[R_{idx}_SEP2]', f'[R_{idx}_SEP3]', f'[R_{idx}_SEP4]'
            name = relations[r]['name']
            relations[r] = {
                'sep1': sep1,  # sep1 to sep4 are used as soft prompts
                'sep2': sep2,
                'sep3': sep3,
                'sep4': sep4,
                'name': name,  # raw name of relations, we do not need new tokens to replace raw names
            }

        return entities, relations

    def resize_tokenizer(self):
        """
        add the new tokens in self.entities and self.relations into the tokenizer of BERT
        :return: a Python Dict, indicating the positions of entities in logtis
        """
        entity_begin_idx = len(self.tokenizer)
        entity_names = [self.entities[e]['name'] for e in self.entities]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_names})
        entity_end_idx = len(self.tokenizer)

        relation_begin_idx = len(self.tokenizer)
        relation_names = [self.relations[r]['sep1'] for r in self.relations]
        relation_names += [self.relations[r]['sep2'] for r in self.relations]
        relation_names += [self.relations[r]['sep3'] for r in self.relations]
        relation_names += [self.relations[r]['sep4'] for r in self.relations]
        relation_names += [self.neighbor_token, self.no_relation_token]
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relation_names})
        relation_end_idx = relation_begin_idx + 4 * len(self.relations) + 2

        return {
            'text_entity_begin_idx': entity_begin_idx,
            'text_entity_end_idx': entity_end_idx,
            'text_relation_begin_idx': relation_begin_idx,
            'text_relation_end_idx': relation_end_idx,
        }

    def get_vocab(self):
        """
        construct the vocab for our KGE module
        :return: two Python Dict
        """
        tokens = ['[PAD]', '[MASK]', '[SEP]', self.no_relation_token]
        entity_names = [e for e in self.entities]
        relation_names = []
        for r in self.relations:
            relation_names += [r, f'{r}_reverse']

        entity_begin_idx = len(tokens)
        entity_end_idx = len(tokens) + len(entity_names)
        relation_begin_idx = len(tokens) + len(entity_names)
        relation_end_idx = len(tokens) + len(entity_names) + len(relation_names)

        tokens = tokens + entity_names + relation_names
        vocab = dict()
        for idx, token in enumerate(tokens):
            vocab[token] = idx

        return vocab, {
            'struc_entity_begin_idx': entity_begin_idx,
            'struc_entity_end_idx': entity_end_idx,
            'struc_relation_begin_idx': relation_begin_idx,
            'struc_relation_end_idx': relation_end_idx,
        }

    def read_lines(self):
        """
        read triplets from  files
        :return: a Python Dict, {train: [], dev: [], test: []}
        """
        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'dev.txt'),
            'test': os.path.join(self.data_path, 'test.txt')
        }

        lines = dict()
        for mode in data_paths:
            data_path = data_paths[mode]
            raw_data = list()

            # 1. read triplets from files
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = str(line).strip().split('\t')
                    raw_data.append((h, r, t))

            # 2. filter triplets which have no textual information
            data = list()
            for h, r, t in raw_data:
                if (h in self.entities) and (t in self.entities) and (r in self.relations):
                    data.append((h, r, t))
            if len(raw_data) > len(data):
                raise ValueError('There are some triplets missing textual information')

            lines[mode] = data

        return lines

    def get_neighbors(self):
        """
        construct neighbor prompts from training dataset
        :return: {entity_id: {text_prompt: [], struc_prompt: []}, ...}
        """
        sep_token = self.tokenizer.sep_token
        mask_token = self.tokenizer.mask_token

        lines = self.lines['train']
        data = {e: {'text_prompt': [], 'struc_prompt': []} for e in self.entities}
        for h, r, t in lines:
            head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
            h_name, r_name, t_name = head['name'], rel['name'], tail['name']
            sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

            # 1. neighbor prompt for predicting head entity
            head_text_prompt = f'{sep1} {mask_token} {sep2} {r_name} {sep3} {t_name} {sep4}'
            head_struc_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
            data[h]['text_prompt'].append(head_text_prompt)
            data[h]['struc_prompt'].append(head_struc_prompt)
            # 2. neighbor prompt for predicting tail entity
            tail_text_prompt = f'{sep1} {h_name} {sep2} {r_name} {sep3} {mask_token} {sep4}'
            tail_struc_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
            data[t]['text_prompt'].append(tail_text_prompt)
            data[t]['struc_prompt'].append(tail_struc_prompt)

        # add a fake neighbor if there is no neighbor for the entity
        for ent in data:
            if len(data[ent]['text_prompt']) == 0:
                h_name = self.entities[ent]['name']
                text_prompt = ' '.join([h_name, sep_token, self.no_relation_token, sep_token, mask_token])
                struc_prompt = [self.vocab[ent], self.vocab[self.no_relation_token], self.vocab[mask_token]]
                data[ent]['text_prompt'].append(text_prompt)
                data[ent]['struc_prompt'].append(struc_prompt)

        return data

    def get_entity_filter(self):
        """
        for given h, r, collect all t
        :return: a Python Dict, {(h, r): [t1, t2, ...]}
        """
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)
        for h, r, t in lines:
            entity_filter[h, r].add(self.entities[t]['token_id'])
            entity_filter[t, r].add(self.entities[h]['token_id'])
        return entity_filter

    def create_examples(self):
        """
        :return: {train: [], dev: [], test: []}
        """
        examples = dict()
        for mode in self.lines:
            data = list()
            lines = self.lines[mode]
            for h, r, t in tqdm(lines, desc=f'[{mode}]构建examples'):
                head_example, tail_example = self.create_one_example(h, r, t)
                data.append(head_example)
                data.append(tail_example)
            examples[mode] = data
        return examples

    def create_one_example(self, h, r, t):
        mask_token = self.tokenizer.mask_token
        sep_token = self.tokenizer.sep_token
        neighbor_token = self.neighbor_token

        head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
        h_name, h_desc = head['name'], head['desc']
        r_name = rel['name']
        t_name, t_desc = tail['name'], tail['desc']
        sep1, sep2, sep3, sep4 = rel['sep1'], rel['sep2'], rel['sep3'], rel['sep4']

        # 1. prepare inputs for nbert
        if self.encode_text:
            if self.add_neighbors:
                text_head_prompt = ' '.join(
                    [sep1, mask_token, sep2, r_name, sep3, t_name, neighbor_token, sep4, t_desc])
                text_tail_prompt = ' '.join(
                    [sep1, h_name, neighbor_token, sep2, r_name, sep3, mask_token, sep4, h_desc])
            else:
                text_head_prompt = ' '.join([sep1, mask_token, sep2, r_name, sep3, t_name, sep4, t_desc])
                text_tail_prompt = ' '.join([sep1, h_name, sep2, r_name, sep3, mask_token, sep4, h_desc])
        else:
            text_head_prompt, text_tail_prompt = None, None
        # 2. prepare inputs for nformer
        if self.encode_struc:
            struc_head_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
            struc_tail_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
        else:
            struc_head_prompt, struc_tail_prompt = None, None
        # 3. get filters
        head_filters = list(self.entity_filter[t, r] - {head['token_id']})
        tail_filters = list(self.entity_filter[h, r] - {tail['token_id']})
        # 4. prepare examples
        head_example = {
            'data_triple': (t, r, h),
            'data_text': (tail["raw_name"], r_name, head['raw_name']),
            'text_prompt': text_head_prompt,
            'struc_prompt': struc_head_prompt,
            'neighbors_label': tail['token_id'],
            'label': head["token_id"],
            'filters': head_filters,
        }
        tail_example = {
            'data_triple': (h, r, t),
            'data_text': (head['raw_name'], r_name, tail['raw_name']),
            'text_prompt': text_tail_prompt,
            'struc_prompt': struc_tail_prompt,
            'neighbors_label': head['token_id'],
            'label': tail["token_id"],
            'filters': tail_filters,
        }

        return head_example, tail_example

    def create_pretrain_examples(self):
        examples = dict()
        for mode in ['train', 'dev', 'test']:
            data = list()
            for h in self.entities.keys():
                name = str(self.entities[h]['name'])
                desc = str(self.entities[h]['desc'])
                desc_tokens = desc.split()

                prompts = [f'The description of {self.tokenizer.mask_token} is that {desc}']
                for i in range(10):
                    begin = random.randint(0, len(desc_tokens))
                    end = min(begin + self.max_seq_length, len(desc_tokens))
                    new_desc = ' '.join(desc_tokens[begin: end])
                    prompts.append(f'The description of {self.tokenizer.mask_token} is that {new_desc}')
                for prompt in prompts:
                    data.append({'prompt': prompt, 'label': self.entities[h]['token_id']})
            examples[mode] = data
        return examples

    def text_batch_encoding(self, inputs):
        encoded_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_data['input_ids'])
        token_type_ids = torch.tensor(encoded_data['token_type_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])
        mask_pos = torch.nonzero(torch.eq(input_ids, self.tokenizer.mask_token_id))

        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'mask_pos': mask_pos}

    def struc_batch_encoding(self, inputs):
        input_ids = torch.tensor(inputs)
        return {'input_ids': input_ids}

    def collate_fn(self, batch_data):
        if self.task == 'pretrain':
            return self.collate_fn_for_pretrain(batch_data)

        # metadata
        data_triple = [data_dit['data_triple'] for data_dit in batch_data]  # [(h, r, t), ...]
        data_text = [data_dit['data_text'] for data_dit in batch_data]  # [(text, text, text), ...]

        text_prompts = [data_dit['text_prompt'] for data_dit in batch_data]  # [string, ...]
        text_data = self.text_batch_encoding(text_prompts) if self.encode_text else None
        struc_prompts = [copy.deepcopy(data_dit['struc_prompt']) for data_dit in batch_data]  # [string, ...]
        struc_data = self.struc_batch_encoding(struc_prompts) if self.encode_struc else None

        if self.add_neighbors:
            batch_text_neighbors = [[] for _ in range(self.neighbor_num)]
            batch_struc_neighbors = [[] for _ in range(self.neighbor_num)]
            for ent, _, _ in data_triple:
                text_neighbors, struc_neighbors = self.neighbors[ent]['text_prompt'], self.neighbors[ent]['struc_prompt']
                idxs = list(range(len(text_neighbors)))
                if len(idxs) >= self.neighbor_num:
                    idxs = random.sample(idxs, self.neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.neighbor_num
                for i, idx in enumerate(idxs):
                    batch_text_neighbors[i].append(text_neighbors[idx])
                    batch_struc_neighbors[i].append(struc_neighbors[idx])
            # neighbor_num * batch_size
            text_neighbors = [self.text_batch_encoding(batch_text_neighbors[i]) for i in range(self.neighbor_num)] \
                if self.encode_text else None
            struc_neighbors = [self.struc_batch_encoding(batch_struc_neighbors[i]) for i in range(self.neighbor_num)] \
                if self.encode_struc else None
        else:
            text_neighbors, struc_neighbors = None, None

        neighbors_labels = torch.tensor([data_dit['neighbors_label']for data_dit in batch_data]) \
            if self.add_neighbors else None
        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'data': data_triple, 'data_text': data_text,
            'text_data': text_data, 'text_neighbors': text_neighbors,
            'struc_data': struc_data, 'struc_neighbors': struc_neighbors,
            'labels': labels, 'filters': filters, 'neighbors_labels': neighbors_labels,
        }

    def collate_fn_for_pretrain(self, batch_data):
        assert self.task == 'pretrain'

        lm_prompts = [data_dit['prompt'] for data_dit in batch_data]  # [string, ...]
        lm_data = self.text_batch_encoding(lm_prompts)

        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])

        return {'text_data': lm_data, 'labels': labels, 'filters': None}

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_dev_dataloader(self):
        dataloader = DataLoader(self.dev_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_test_dataloader(self):
        dataloader = DataLoader(self.test_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_tokenizer(self):
        return self.tokenizer


if __name__ == '__main__':
    pass
    # data_path = os.path.join(os.getcwd(), '../../dataset/wn18rr/support')
    # entity_path = os.path.join(data_path, 'entity2text.txt')
    # relation_path = os.path.join(data_path, 'relation2text.txt')
    # entity_dict = dict()
    # relation_dict = dict()
    #
    # with open(entity_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         tokens = str(line).strip().split('\t')
    #         assert len(tokens) == 2
    #         eid, desc = tokens
    #         name = desc.split(',')[0]
    #         entity_dict[eid] = {'name': name, 'desc': desc}
    # with open(os.path.join(data_path, 'entity.json'), 'w', encoding='utf-8') as f:
    #     json.dump(entity_dict, f)
    #
    # with open(relation_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         tokens = str(line).strip().split('\t')
    #         assert len(tokens) == 2
    #         rid, name = tokens
    #         relation_dict[rid] = {'name': name, desc: ''}
    # with open(os.path.join(data_path, 'relation.json'), 'w', encoding='utf-8') as f:
    #     json.dump(relation_dict, f)
