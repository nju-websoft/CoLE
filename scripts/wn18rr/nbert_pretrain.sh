# add_neighbors neighbor_num are not used  for pretrain

python train_nbert.py \
   --task pretrain \
   --model_path checkpoints/bert-base-cased \
   --epoch 20 \
   --batch_size 256 \
   --device cuda:1 \
   --dataset wn18rr \
   --max_seq_length 32 \
   --lm_lr 1e-4 \
   --lm_label_smoothing 0.8 \
   --num_workers 64 \
   --pin_memory True