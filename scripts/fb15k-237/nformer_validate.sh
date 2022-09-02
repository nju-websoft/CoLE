# output/fb15k-237/N-Former/20220823_202921/avg.bin
# output/fb15k-237/CoLE/20220829_123331/nformer.bin

python train_nformer.py \
   --task validate \
   --model_path output/fb15k-237/N-Former/20220823_202921/avg.bin \
   --batch_size 2048 \
   --device cuda:2 \
   --dataset fb15k-237 \
   --num_workers 32 \
   --pin_memory True