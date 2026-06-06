# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SwinIR is an **inference-only** PyTorch implementation of image restoration using Swin Transformer. Training code lives in the separate [KAIR](https://github.com/cszn/KAIR) repo.

There is no test suite, CI, linting, or setup.py. Dependencies are listed in `requirements.txt` (modern) and `cog.yaml` (Replicate deployment, older pinned versions).

## Architecture

`models/network_swinir.py` contains the full model in a single file:

- **Top-level**: `SwinIR` — three-stage pipeline: shallow feature extraction → deep feature extraction → image reconstruction.
- **Deep feature extraction**: A sequence of `RSTB` (Residual Swin Transformer Block) modules. Each RSTB wraps a `BasicLayer` (multiple `SwinTransformerBlock` instances) with a conv-based residual connection.
- **Window attention**: `WindowAttention` uses relative position bias. `window_partition`/`window_reverse` handle the shifted-window token shuffling.
- **Reconstruction** (controlled by `upsampler`):
  - `pixelshuffle` — classical SR (conv → PixelShuffle chain)
  - `pixelshuffledirect` — lightweight SR (single conv + PixelShuffle, fewer params)
  - `nearest+conv` — real-world SR (nearest-neighbor upsampling + conv refinement, fewer block artifacts)
  - empty/None — denoising and JPEG CAR (no upscaling, just conv_last)

Key model parameters are task-specific and set in `main_test_swinir.py:define_model()`.

## Common commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Run inference on a single image (local script, no Cog dependency):
```bash
python run_predict.py --task real_sr --scale 4 --model_path <path.pth> --input <image.jpg>
```

Run batch evaluation with PSNR/SSIM metrics:
```bash
python main_test_swinir.py --task <task> --model_path <path.pth> --folder_gt <dir>
```

If `--model_path` doesn't exist on disk, the script auto-downloads from GitHub Releases into `model_zoo/swinir/`.

For large images, add `--tile 400` to process in tiles and avoid OOM. Tile size must be a multiple of `window_size`.

## Tasks and model configurations

| `--task` | Scale | `in_chans` | `img_size` | `window_size` | `img_range` | `upsampler` | `embed_dim` | `depths` |
|---|---|---|---|---|---|---|---|---|
| `classical_sr` | 2/3/4/8 | 3 | 48 or 64 | 8 | 1.0 | `pixelshuffle` | 180 | 6×6 |
| `lightweight_sr` | 2/3/4 | 3 | 64 | 8 | 1.0 | `pixelshuffledirect` | 60 | 4×6 |
| `real_sr` (M) | 4 | 3 | 64 | 8 | 1.0 | `nearest+conv` | 180 | 6×6 |
| `real_sr` (L) | 4 | 3 | 64 | 8 | 1.0 | `nearest+conv` | 240 | 9×8 |
| `gray_dn` | 1 | 1 | 128 | 8 | 1.0 | `""` | 180 | 6×6 |
| `color_dn` | 1 | 3 | 128 | 8 | 1.0 | `""` | 180 | 6×6 |
| `jpeg_car` | 1 | 1 | 126 | 7 | 255.0 | `""` | 180 | 6×6 |
| `color_jpeg_car` | 1 | 3 | 126 | 7 | 255.0 | `""` | 180 | 6×6 |

## Checkpoint loading

Pretrained weights are loaded in `define_model()`. Most tasks use key `'params'`, but `real_sr` uses `'params_ema'`. Loading with the wrong key will fail.

## Shell scripts

- `run_real_sr.sh` — batch real-world SR on `testsets/RealSRSet+5images`
- `run_predict.sh <image>` — single-image real-world SR (wraps `run_predict.py`)
- `download-weights.sh` — downloads weights into `experiments/pretrained_models/` (used by the Cog predictor)

## Device support

Priority order: CUDA → MPS (Apple Silicon) → CPU. Both `main_test_swinir.py` and `run_predict.py` follow this pattern.

## Input/output conventions

- Input images are read as BGR via OpenCV, normalized to [0, 1] float32, then transposed to CHW-RGB before being fed to the model as NCHW tensors.
- Inputs are reflection-padded to multiples of `window_size` before inference, then cropped back.
- Output is clamped to [0, 1], converted back to HWC-BGR, scaled to [0, 255] uint8, and saved via `cv2.imwrite`.

## 交互提示

- 每完成一项工作后,提示爸爸说你的工作已经完成