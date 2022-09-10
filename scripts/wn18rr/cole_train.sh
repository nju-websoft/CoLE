# output/wn18rr/N-BERT/20220904_124646/nbert
# output/wn18rr/N-Former/20220906_103935/avg.bin

python train_cole.py \
   --task train \
   --nbert_path output/wn18rr/N-BERT/20220904_124646/nbert \
   --nformer_path output/wn18rr/N-Former/20220906_103935/avg.bin \
   --alpha 0.5 \
   --beta 0.5 \
   --epoch 30 \
   --batch_size 256 \
   --device cuda:2 \
   --dataset wn18rr \
   --max_seq_length 64 \
   --add_neighbors \
   --neighbor_num 1 \
   --lm_lr 5e-5 \
   --lm_label_smoothing 0.8 \
   --kge_lr 5e-5 \
   --kge_label_smoothing 0.8 \
   --num_workers 32 \
   --pin_memory True