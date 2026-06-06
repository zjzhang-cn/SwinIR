# SwinIR 推理脚本使用说明

`run_predict.py` 是 SwinIR 项目的**独立推理脚本**，整合了 `main_test_swinir.py` 的全部功能，无需 Cog 依赖，可直接命令行调用。

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

核心依赖：`torch >= 1.8.0`、`timm >= 0.4.12`、`opencv-python`、`requests`

### 模型下载

如果 `--model_path` 指定的模型文件不存在，脚本会自动从 SwinIR 的 GitHub Releases 下载。各任务对应的预训练模型见[官方 Releases](https://github.com/JingyunLiang/SwinIR/releases)。

## 支持的任务

| 任务类型 | `--task` | 说明 | 关键参数 |
|---|---|---|---|
| 经典图像超分 | `classical_sr` | bicubic 下采样的图像 SR | `--scale 2/3/4/8`、`--training_patch_size 48/64` |
| 轻量图像超分 | `lightweight_sr` | 更少参数的轻量 SR | `--scale 2/3/4` |
| 真实世界超分 | `real_sr` | 真实低质图像 SR（减少块效应） | `--scale 4`、`--large_model` |
| 灰度图像去噪 | `gray_dn` | 去除高斯噪声 | `--noise 15/25/50` |
| 彩色图像去噪 | `color_dn` | 去除高斯噪声 | `--noise 15/25/50` |
| 灰度 JPEG 去伪影 | `jpeg_car` | 去除 JPEG 压缩伪影 | `--jpeg 10/20/30/40` |
| 彩色 JPEG 去伪影 | `color_jpeg_car` | 去除 JPEG 压缩伪影 | `--jpeg 10/20/30/40` |

## 两种运行模式

### 模式一：单张图像推理

传入 `--input` 参数，直接对一张图像进行增强并保存结果。

```bash
# 真实世界超分（4x）
python run_predict.py \
    --task real_sr --scale 4 \
    --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
    --input photo.jpg

# 彩色去噪（noise=25）
python run_predict.py \
    --task color_dn --noise 25 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth \
    --input noisy_photo.jpg

# 灰度去噪（noise=15），灰度图输入
python run_predict.py \
    --task gray_dn --noise 15 \
    --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth \
    --input gray_noisy.png

# 灰度 JPEG 去伪影
python run_predict.py \
    --task jpeg_car --jpeg 20 \
    --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth \
    --input compressed.jpg

# 指定输出路径（默认：输入文件名_SwinIR.png）
python run_predict.py \
    --task real_sr --scale 4 \
    --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
    --input photo.jpg --output result.jpg
```

### 模式二：批量文件夹评估

传入 `--folder_gt`（和其他必要参数），脚本遍历文件夹内所有图像，逐张推理并计算 PSNR/SSIM 指标。

```bash
# 经典 SR 评估（x2，DIV2K 训练，patch_size=48）
python run_predict.py \
    --task classical_sr --scale 2 --training_patch_size 48 \
    --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth \
    --folder_lq testsets/Set5/LR_bicubic/X2 \
    --folder_gt testsets/Set5/HR

# 经典 SR 评估（x2，DIV2K+Flickr2K 训练，patch_size=64）
python run_predict.py \
    --task classical_sr --scale 2 --training_patch_size 64 \
    --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth \
    --folder_lq testsets/Set5/LR_bicubic/X2 \
    --folder_gt testsets/Set5/HR

# 轻量 SR 评估
python run_predict.py \
    --task lightweight_sr --scale 2 \
    --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth \
    --folder_lq testsets/Set5/LR_bicubic/X2 \
    --folder_gt testsets/Set5/HR

# 真实世界 SR（无 GT，不计算指标）
python run_predict.py \
    --task real_sr --scale 4 \
    --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
    --folder_lq testsets/RealSRSet+5images

# 彩色去噪评估（在线加噪，固定 seed=0 保证可复现）
python run_predict.py \
    --task color_dn --noise 15 \
    --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
    --folder_gt testsets/McMaster

# 灰度去噪评估
python run_predict.py \
    --task gray_dn --noise 15 \
    --model_path model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth \
    --folder_gt testsets/Set12

# JPEG 去伪影评估
python run_predict.py \
    --task jpeg_car --jpeg 40 \
    --model_path model_zoo/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth \
    --folder_gt testsets/classic5
```

## 大图分块推理

对于显存不足的大分辨率图像，使用 `--tile` 参数启用分块推理。

```bash
python run_predict.py \
    --task real_sr --scale 4 \
    --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
    --input huge_panorama.jpg --tile 400
```

分块推理原理：将图像按 `--tile` 大小切分为有重叠的块 → 逐块送入模型 → 按加权平均拼回完整图像（重叠区域平滑融合）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tile` | `None` | 分块大小，需为 `window_size`（7 或 8）的倍数，建议 400 |
| `--tile_overlap` | `32` | 相邻块之间的重叠像素数，值越大拼接越平滑但越慢 |

## 完整参数列表

| 参数 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `--task` | str | 否 | 任务类型，默认 `real_sr` |
| `--scale` | int | 否 | 超分倍数：1/2/3/4/8，去噪/JPEG 任务设 1 |
| `--noise` | int | 否 | 噪声等级：15/25/50（仅去噪任务） |
| `--jpeg` | int | 否 | JPEG 质量因子：10/20/30/40（仅 JPEG CAR） |
| `--training_patch_size` | int | 否 | 训练 patch 大小：48/64（仅 classical_sr，不影响实际推理） |
| `--large_model` | flag | 否 | 使用 SwinIR-L 大模型（仅 real_sr） |
| `--model_path` | str | **是** | 预训练模型路径，不存在时自动下载 |
| `--input` | str | 单张模式 | 单张输入图像路径 |
| `--folder_lq` | str | 批量模式 | 低质量测试图像目录 |
| `--folder_gt` | str | 批量模式 | 真实高质量图像目录（用于 PSNR/SSIM） |
| `--output` | str | 否 | 输出路径（单张模式，默认填入文件名_SwinIR.png） |
| `--tile` | int | 否 | 分块大小（不设则整图推理） |
| `--tile_overlap` | int | 否 | 分块重叠像素，默认 32 |

## 模型配置速查

| `--task` | scale | in_chans | img_size | window_size | img_range | upsampler | embed_dim | depths |
|---|---|---|---|---|---|---|---|---|
| `classical_sr` | 2/3/4/8 | 3 | 48/64 | 8 | 1.0 | `pixelshuffle` | 180 | 6×6 |
| `lightweight_sr` | 2/3/4 | 3 | 64 | 8 | 1.0 | `pixelshuffledirect` | 60 | 4×6 |
| `real_sr` (M) | 4 | 3 | 64 | 8 | 1.0 | `nearest+conv` | 180 | 6×6 |
| `real_sr` (L) | 4 | 3 | 64 | 8 | 1.0 | `nearest+conv` | 240 | 9×6 |
| `gray_dn` | 1 | 1 | 128 | 8 | 1.0 | `""` | 180 | 6×6 |
| `color_dn` | 1 | 3 | 128 | 8 | 1.0 | `""` | 180 | 6×6 |
| `jpeg_car` | 1 | 1 | 126 | 7 | 255.0 | `""` | 180 | 6×6 |
| `color_jpeg_car` | 1 | 3 | 126 | 7 | 255.0 | `""` | 180 | 6×6 |

## 注意事项

1. **设备支持**：自动检测 CUDA → MPS (Apple Silicon) → CPU，MPS 后端结果可能与 CUDA 略有差异。
2. **检查点键名**：大多数模型使用 `'params'` 键加载权重，`real_sr` 使用 `'params_ema'`。脚本会自动处理。
3. **窗口填充**：输入图像会被反射填充至 `window_size` 的整数倍，推理后裁剪回原始尺寸。
4. **评估边界裁剪**：classical_sr 和 lightweight_sr 评估 PSNR/SSIM 时会裁剪 `scale` 像素的边界。
5. **单张模式 vs 批量模式**：通过 `--input` 参数自动切换。有 `--input` 即单张模式，否则为批量模式（需要 `--folder_gt`）。
6. **JPEG CAR 的 window_size**：因为 JPEG 使用 8×8 块编码，window_size 设为 7（而非 8）以更好地对齐块边界。
7. **去噪的噪声**：灰度/彩色去噪任务在批量模式下通过 `np.random.seed(0)` 固定噪声种子，确保评估结果可复现。单张模式下直接读取输入图像，不做在线加噪。
