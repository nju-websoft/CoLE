# epoch lm_lr lm_label_smoothing are not used  for validate
# output/fb15k-237/N-BERT/20220827_224256/nbert
# output/fb15k-237/CoLE/20220829_123331/nbert

python train_nbert.py \
   --task validate \
   --model_path $MODEL_PATH \
   --batch_size 256 \
   --device cuda:1 \
   --dataset wn18rr \
   --max_seq_length 64 \
   --add_neighbors \
   --neighbor_num 3 \
   --num_workers 32 \
   --pin_memory True