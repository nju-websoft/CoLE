# output/fb15k-237/N-BERT/20220824_132950/nbert
# output/fb15k-237/N-Former/20220823_202921/avg.bin
# output/fb15k-237/CoLE/20220829_123331/nbert
# output/fb15k-237/CoLE/20220829_123331/nformer.bin

python train_cole.py \
   --task validate \
   --nbert_path output/fb15k-237/CoLE/20220829_123331/nbert \
   --nformer_path output/fb15k-237/CoLE/20220829_123331/nformer.bin \
   --batch_size 256 \
   --device cuda:2 \
   --dataset fb15k-237 \
   --max_seq_length 64 \
   --add_neighbors \
   --neighbor_num 3 \
   --num_workers 32 \
   --pin_memory True