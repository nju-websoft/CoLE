python train_cole.py \
   --task train \
   --nbert_path output/fb15k-237/N-BERT/20220824_132950/nbert \
   --nformer_path output/fb15k-237/N-Former/20220823_202921/avg.bin \
   --alpha 0.5 \
   --beta 0.6 \
   --epoch 20 \
   --batch_size 256 \
   --device cuda:2 \
   --dataset fb15k-237 \
   --max_seq_length 64 \
   --add_neighbors \
   --neighbor_num 3 \
   --lm_lr 1e-5 \
   --lm_label_smoothing 0.8 \
   --kge_lr 5e-5 \
   --kge_label_smoothing 0.8 \
   --num_workers 32 \
   --pin_memory True