#!/bin/bash
PORT=${1:-8000}
TASK=${2:-real_sr}
SCALE=${3:-4}
MODEL=${4:-model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth}

python web_server.py \
    --task "$TASK" \
    --scale "$SCALE" \
    --model_path "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0
