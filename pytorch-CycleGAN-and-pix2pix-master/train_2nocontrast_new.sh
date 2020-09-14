#!/usr/bin/env bash
python train.py \
    --batch_size 12 \
    --gpu_ids 0,1,2,3 \
    --name contrast2no_new \
    --dataroot datasets/contrast2no_new \
    --input_nc 1 \
    --output_nc 1 \
    --load_size 512 \
    --crop_size 512 \
    --display_winsize 512 \
    --display_id 0;

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/24
