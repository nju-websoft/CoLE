import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .knowformer_encoder import Embeddings, Encoder, truncated_normal_init, norm_layer_init
import time


class Knowformer(nn.Module):
    def __init__(self, config):
        super(Knowformer, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._input_dropout_prob = config['input_dropout_prob']
        self._attention_dropout_prob = config['attention_dropout_prob']
        self._hidden_dropout_prob = config['hidden_dropout_prob']
        self._residual_dropout_prob = config['residual_dropout_prob']
        self._context_dropout_prob = config['context_dropout_prob']
        self._initializer_range = config['initializer_range']
        self._intermediate_size = config['intermediate_size']

        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']

        self.ele_embedding = Embeddings(self._emb_size, self._voc_size, self._initializer_range)

        self.triple_encoder = Encoder(config)
        self.context_encoder = Encoder(config)

        # 对输入的准备
        self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
        self.context_dropout_layer = nn.Dropout(p=self._context_dropout_prob)

    def __forward_triples(self, triple_ids, context_emb=None, encoder_type="triple"):
        # convert token id to embedding
        emb_out = self.ele_embedding(triple_ids)  # (batch_size, 3, embed_size)

        # merge context_emb into emb_out
        if context_emb is not None:
            context_emb = self.context_dropout_layer(context_emb)
            emb_out[:, 0, :] = (emb_out[:, 0, :] + context_emb) / 2

        emb_out = self.input_dropout_layer(emb_out)
        encoder = self.triple_encoder if encoder_type == "triple" else self.context_encoder
        emb_out = encoder(emb_out, mask=None)  # (batch_size, 3, embed_size)
        return emb_out

    def __process_mask_feat(self, mask_feat):
        return torch.matmul(mask_feat, self.ele_embedding.lut.weight.transpose(0, 1))

    def forward(self, src_ids, window_ids=None, double_encoder=False):
        # src_ids: (batch_size, seq_size, 1)
        # window_ids: (batch_size, seq_size) * neighbor_num

        # 1. do not use embeddings from neighbors
        seq_emb_out = self.__forward_triples(src_ids, context_emb=None)
        mask_emb = seq_emb_out[:, 2, :]  # (batch_size, embed_size)
        logits_from_triplets = self.__process_mask_feat(mask_emb)  # (batch_size, vocab_size)

        if window_ids is None:
            return {'without_neighbors': logits_from_triplets, 'with_neighbors': None, 'neighbors': None}

        # 2. encode neighboring triplets
        logits_from_neighbors = []
        embeds_from_neighbors = []
        for i in range(len(window_ids)):
            if double_encoder:
                seq_emb_out = self.__forward_triples(window_ids[i], context_emb=None, encoder_type='context')
            else:
                seq_emb_out = self.__forward_triples(window_ids[i], context_emb=None, encoder_type='triple')
            mask_emb = seq_emb_out[:, 2, :]
            logits = self.__process_mask_feat(mask_emb)

            embeds_from_neighbors.append(mask_emb)
            logits_from_neighbors.append(logits)
        # get embeddings from neighboring triplets by averaging
        context_embeds = torch.stack(embeds_from_neighbors, dim=0)  # (neighbor_num, batch_size, 768)
        context_embeds = torch.mean(context_embeds, dim=0)

        # 3. leverage both the triplet and neighboring triplets
        seq_emb_out = self.__forward_triples(src_ids, context_emb=context_embeds)
        mask_embed = seq_emb_out[:, 2, :]
        logits_from_both = self.__process_mask_feat(mask_embed)

        return {
            'without_neighbors': logits_from_triplets,
            'with_neighbors': logits_from_both,
            'neighbors': logits_from_neighbors,
        }

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers.models.bert.modeling_bert import BertPreTrainedModel
# from transformers import BertConfig
# from code.models.knowformer_encoder import Embeddings, Encoder, truncated_normal_init, norm_layer_init
#
#
# '''
# 基于自我实现的transformer架构进行link prediction
# '''
# # codes form LGY
# class KgeBertModel(BertPreTrainedModel):
#     def __init__(self, config: BertConfig):
#         super(KgeBertModel, self).__init__(config)
#         self.config_dict = self._convert_config2dict(config)
#
#         self._emb_size = self.config_dict['hidden_size']
#         self._input_dropout_prob = self.config_dict['input_dropout_prob']
#         self._context_dropout_prob = self.config_dict['context_dropout_prob']
#         self._initializer_range = self.config_dict['initializer_range']
#         self._voc_size = self.config_dict['vocab_size']
#
#         self.ele_embedding = Embeddings(self._emb_size, self._voc_size, self._initializer_range)
#
#         self.triple_encoder = Encoder(self.config_dict)
#         self.context_encoder = Encoder(self.config_dict)
#
#         # 对输入的准备
#         self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
#         self.input_norm_layer = nn.LayerNorm(self._emb_size)
#         norm_layer_init(self.input_norm_layer)
#
#         self.context_dropout_layer = nn.Dropout(p=self._context_dropout_prob)
#
#         # 对输出的变换
#         self.output_fc_entity = nn.Linear(self._emb_size, self._emb_size)
#         truncated_normal_init(self.output_fc_entity, [self._emb_size, self._emb_size], self._initializer_range)
#         self.output_fc_act = nn.functional.gelu
#         self.output_norm_layer_entity = nn.LayerNorm(self._emb_size)
#         norm_layer_init(self.output_norm_layer_entity)
#         self.bias = torch.nn.Parameter(torch.zeros(self._voc_size))
#
#     def _convert_config2dict(self, config: BertConfig):
#         config_dict = {
#             'hidden_size': config.hidden_size,
#             'num_hidden_layers': config.num_hidden_layers,
#             'num_attention_heads': config.num_attention_heads,
#             'input_dropout_prob': config.input_dropout_prob,
#             'attention_dropout_prob': config.attention_dropout_prob,
#             'hidden_dropout_prob': config.hidden_dropout_prob,
#             'residual_dropout_prob': config.residual_dropout_prob,
#             'context_dropout_prob': config.context_dropout_prob,
#             'initializer_range': config.initializer_range,
#             'intermediate_size': config.intermediate_size,
#             'vocab_size': config.vocab_size,
#             'num_relations': config.num_relations,
#         }
#         return config_dict
#
#     def __forward_triples(self, triple_ids, context_emb=None, encoder_type="triple"):
#         # 对输入的准备
#         emb_out = self.ele_embedding(triple_ids)
#         # 将context_emb 与 emb_out[0, :, :] 进行融合
#         if context_emb is not None:
#             context_emb = self.context_dropout_layer(context_emb)
#             context_emb = self.output_fc_act(context_emb)  # 这里用不用gelu差不多
#             emb_out[:, 0, :] = (emb_out[:, 0, :] + context_emb) / 2
#
#         emb_out = self.input_dropout_layer(emb_out)
#         if encoder_type == "triple":
#             encoder = self.triple_encoder
#         else:
#             encoder = self.context_encoder
#         emb_out = encoder(emb_out, mask=None)
#         return emb_out
#
#     def __process_mask_feat(self, mask_feat):
#         return torch.matmul(mask_feat, self.ele_embedding.lut.weight.transpose(0, 1))
#
#     def forward(self, src_ids, window_ids):
#         # src_ids.shape (batch_size, seq_size)
#         # window_ids.shape (batch_size * window_size, seq_size)
#         window_emb_out = self.__forward_triples(window_ids, encoder_type='context')
#         window_mask_feat = window_emb_out[:, 2, :]
#         window_predict = self.__process_mask_feat(window_mask_feat)  # 邻居窗口的预测
#
#         context_emb = torch.reshape(window_mask_feat, (src_ids.shape[0], window_ids.shape[0] // src_ids.shape[0], -1))
#         context_emb = torch.mean(context_emb, dim=1)
#
#         mixed_src_emb_out = self.__forward_triples(src_ids, context_emb)
#         mixed_src_mask_feat = mixed_src_emb_out[:, 2, :]
#         mixed_src_predict = self.__process_mask_feat(mixed_src_mask_feat)  # 融入邻居信息后的预测
#
#         origin_src_emb_out = self.__forward_triples(src_ids)
#         origin_src_mask_feat = origin_src_emb_out[:, 2, :]
#         origin_src_predict = self.__process_mask_feat(origin_src_mask_feat)  # 不看邻居信息的预测
#
#         return origin_src_predict, mixed_src_predict, window_predict,
#
