#!/bin/bash
python main_test_swinir.py \
    --task real_sr \
    --scale 4 \
    --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
    --folder_lq testsets/RealSRSet+5images
