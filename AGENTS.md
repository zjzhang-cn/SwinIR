# AGENTS.md

## Repository overview
- This is **inference-only** code for SwinIR. Training code lives in the separate [KAIR](https://github.com/cszn/KAIR) repo.
- No `requirements.txt`, `setup.py`, `pyproject.toml`, CI, linting, or tests exist. Dependencies are only documented in `cog.yaml` (Python 3.8, torch 1.8.0, timm 0.4.12, opencv-python).

## Entrypoints
- `main_test_swinir.py` — batch evaluation script that runs inference and computes PSNR/SSIM metrics.
- `run_predict.py` — single-image inference (no Cog dependency), wraps `define_model`/`setup` from `main_test_swinir.py`.
- `predict.py` — Cog web demo predictor (Replicate), references `experiments/pretrained_models/`.
- `models/network_swinir.py` — single-file model architecture (SwinIR class + all submodules).
- `utils/util_calculate_psnr_ssim.py` — metric utilities.

## Running inference
```bash
python main_test_swinir.py --task <task> --model_path <path> [--folder_lq <dir>] [--folder_gt <dir>] [--scale N] [--noise N] [--jpeg N] [--tile 400]
```
- If `--model_path` doesn't exist on disk, the script auto-downloads from GitHub Releases into `model_zoo/swinir/`.
- For large images, add `--tile 400` to avoid OOM (tile size must be a multiple of `window_size`).

## Key conventions and gotchas
- **Device priority**: CUDA → MPS (Apple Silicon) → CPU. Set in both `main_test_swinir.py` and `run_predict.py`. Note that Apple Silicon MPS backend may produce slightly different results from CUDA.
- **Checkpoint key**: most models use `'params'`, but `real_sr` uses `'params_ema'`. Loading with the wrong key errors. Check `define_model()` for which key each task uses.
- **Window padding**: inputs are auto-padded to multiples of `window_size` via flip padding. `window_size=8` for all tasks except JPEG CAR (`window_size=7`).
- **Image range**: `img_range=1.0` for most tasks, `img_range=255.0` for JPEG CAR tasks.
- **`--training_patch_size`** does NOT affect inference patching; it only selects the correct pretrained model variant (48 vs 64 from Table 2 in the paper).
- **Model zoo paths differ**: `main_test_swinir.py` loads from `model_zoo/swinir/`. `predict.py` and `download-weights.sh` use `experiments/pretrained_models/`.

## Tasks
| `--task` | Description | Extra flags |
|---|---|---|
| `classical_sr` | Bicubic image SR | `--scale 2/3/4/8`, `--training_patch_size 48/64` |
| `lightweight_sr` | Lightweight image SR | `--scale 2/3/4` |
| `real_sr` | Real-world image SR | `--scale 4`, `--large_model` (optional) |
| `gray_dn` | Grayscale denoising | `--noise 15/25/50` |
| `color_dn` | Color denoising | `--noise 15/25/50` |
| `jpeg_car` | Grayscale JPEG CAR | `--jpeg 10/20/30/40` |
| `color_jpeg_car` | Color JPEG CAR | `--jpeg 10/20/30/40` |
