# output/wn18rr/N-Former/20220906_103935/
# output/fb15k-237/CoLE/20220829_123331/nformer.bin

python train_nformer.py \
   --task validate \
   --model_path output/wn18rr/N-Former/20220906_103935/avg.bin \
   --batch_size 1024 \
   --device cuda:2 \
   --dataset wn18rr \
   --num_workers 32 \
   --pin_memory True