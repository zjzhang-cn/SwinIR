#!/bin/bash
INPUT=$1
FILENAME=$(basename "$INPUT")
NAME="${FILENAME%.*}"

python run_predict.py \
    --task real_sr \
    --scale 4 \
    --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
    --input "$INPUT" \
    --output "out-${NAME}.png" \
    --tile 128 \
    --tile_overlap 32
