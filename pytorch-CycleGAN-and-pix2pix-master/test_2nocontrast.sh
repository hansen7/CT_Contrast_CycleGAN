#!/usr/bin/env bash

cp ./checkpoints/contrast2no_new/latest_net_G_A.pth ./checkpoints/contrast2no_new/latest_net_G.pth

python test.py \
    --batch_size 9 \
    --gpu_ids 6,7 \
    --name contrast2no_new \
    --dataroot datasets/contrast2no_new/testA_new \
    --results_dir results/contrast2no_new_GenA \
    --input_nc 1 \
    --output_nc 1 \
    --load_size 512 \
    --crop_size 512 \
    --model test \
    --no_dropout;

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/24
# train/testA: contrast images
# train/testB: no contrast images
