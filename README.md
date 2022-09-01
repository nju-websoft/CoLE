# I Know What You Do not Konw: Knowledge Graph Embedding via Co-distillation Learning

> Knowledge graph (KG) embedding seeks to learn vector representations for entities and relations. Conventional models reason over graph structures, but they suffer from the issues of graph incompleteness and long-tail entities. Recent studies have used pre-trained language models to learn embeddings based on the textual information of entities and relations, but they cannot take advantage of graph structures. In the paper, we show empirically that these two kinds of features are complementary for KG embedding. To this end, we propose CoLE, a **Co**-distillation **L**earning method for KG **E**mbedding that exploits the complementarity of graph structures and text information. Its graph embedding model employs Transformer to reconstruct the representation of an entity from its neighborhood subgraph. Its text embedding model uses a pretrained language model to generate entity representations from the soft prompts of their names, descriptions, and relational neighbors. To let the two model promote each other, we propose co-distillation learning that allows them to distill selective knowledge from each other’s prediction logits. In our co-distillation learning, each model serves as both a teacher and a student. Experiments on benchmark datasets demonstrate that the two models outperform their related baselines, and the ensemble method CoLE with co-distillation learning advances the state-of-the-art of KG embedding.

## Dependencies

- pytorch==1.10.2
- transformers==4.11.3
- contiguous-params==1.0.0


## Running
Create folder `checkpoints` and put the pre-trained [BERT](https://huggingface.co/bert-base-cased/tree/main) model into `checkpoints/bert-base-cased`.

Take FB15k-237 for example, fine-tune a BERT model based on the descriptions of entities; and put the fine-tuned model into `checkpoints/fb15k-237/bert-pretrained`.
```
$ python train_nbert.py --task pretrain --dataset fb15k-237 --device cuda:0 --lm_lr 1e-4
```

Train N-BERT:
```
$ python train_nbert.py --task train ---dataset fb15k-237 --device cuda:0 --lm_lr 5e-5 --add_neighbors
```

Train N-Former:
```
$ python train_nformer.py --task train ---dataset fb15k-237 --device cuda:0 --lm_lr 5e-5 --add_neighbors
```

Train CoLE:
```
$ python train_cole.py --task train ---dataset fb15k-237 --device cuda:0 --alpha 0.5 --beta 0.5
```

> If you have any difficulty or question in running code and reproducing experimental results, please email to yliu20.nju@gmail.com.

## Citation

```
@inproceedings{CoLE,
  title     = {I Know What You Do not Konw: Knowledge Graph Embedding via Co-distillation Learning},
  author    = {Yang Liu and 
               Zequn Sun and 
               Guangyao Li and 
               Wei Hu},
  booktitle = {CIKM},
  year      = {2022}
}
```
