#!/usr/bin/env bash
python test.py \
    --batch_size 9 \
    --gpu_ids 0,1,2,3 \
    --name contrast2no \
    --dataroot datasets/contrast2no/testA \
    --input_nc 1 \
    --output_nc 1 \
    --load_size 512 \
    --crop_size 512 \
    --display_winsize 512 \
    --display_id 0 \
    --model test \
    --no_dropout;

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/24
# train/testA: contrast images
# train/testB: no contrast images