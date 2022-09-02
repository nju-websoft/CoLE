# add_neighbors neighbor_num are not used  for pretrain

python train_nbert.py \
   --task pretrain \
   --model_path checkpoints/bert-base-cased \
   --epoch 20 \
   --batch_size 256 \
   --device cuda:1 \
   --dataset fb15k-237 \
   --max_seq_length 64 \
   --lm_lr 1e-4 \
   --lm_label_smoothing 0.8 \
   --num_workers 32 \
   --pin_memory True