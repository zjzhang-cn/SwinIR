#!/usr/bin/env python
"""
SwinIR 图像推理脚本（完全独立，整合 main_test_swinir.py 全部功能）
=====================================================================

本脚本自包含 SwinIR 图像恢复所需的全部逻辑：模型构建、权重加载、图像预处理、模型推理、
后处理保存及 PSNR/SSIM 评估。无需 Cog 依赖，可直接在命令行调用。

支持两种运行模式：
  1. 单张图像推理（--input  + --model_path）
     → 读取一张低质量图像，输出增强后的高质量图像，不计算 PSNR/SSIM。

  2. 批量文件夹评估（--folder_gt + --model_path）
     → 遍历文件夹内所有图像，逐张推理并计算 PSNR/SSIM 指标（如有 GT），
       结果保存到 results/ 目录。

模型架构说明（models/network_swinir.py 中的 SwinIR 类）：
  SwinIR 是一个三阶段图像恢复模型：
    Stage 1 — 浅层特征提取：一个 3×3 卷积（conv_first）提取浅层特征
    Stage 2 — 深层特征提取：多个 RSTB（Residual Swin Transformer Block）
                 每个 RSTB 包含多个 Swin Transformer block + 卷积残差连接
    Stage 3 — 图像重建：根据任务类型使用不同的 upsampler 进行高质量重建

不同任务通过 upsampler 参数区分重建方式：
  - pixelshuffle      经典 SR：多级 Conv2d + PixelShuffle 上采样
  - pixelshuffledirect 轻量 SR：单级 Conv2d + PixelShuffle（参数更少，速度快）
  - nearest+conv      真实世界 SR：最近邻插值上采样 + Conv 细化（减少块效应）
  - 空字符串 ""       去噪 / JPEG CAR：不上采样，仅一个 conv_last 输出

本脚本支持的全部 7 种任务及对应论文编号：
  001 - classical_sr    经典 bicubic 下采样图像超分辨率
  002 - lightweight_sr  轻量图像超分辨率（参数量更小）
  003 - real_sr         真实世界图像超分辨率（未知退化模型）
  004 - gray_dn         灰度图像去噪（高斯噪声 σ=15/25/50）
  005 - color_dn        彩色图像去噪（高斯噪声 σ=15/25/50）
  006 - jpeg_car        灰度 JPEG 压缩伪影去除（质量因子 Q=10/20/30/40）
  006 - color_jpeg_car  彩色 JPEG 压缩伪影去除（质量因子 Q=10/20/30/40）

用法示例：
  # 单张图像推理（仅输出增强后的图像）
  python run_predict.py --task real_sr --scale 4 \\
      --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \\
      --input photo.jpg

  # 批量文件夹评估（输出图像 + PSNR/SSIM 指标）
  python run_predict.py --task classical_sr --scale 2 --training_patch_size 48 \\
      --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth \\
      --folder_gt testsets/Set5/HR --folder_lq testsets/Set5/LR_bicubic/X2

  # 大图分块推理（避免显存不足，tile 需为 window_size 的倍数）
  python run_predict.py --task real_sr --scale 4 \\
      --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \\
      --input huge_panorama.jpg --tile 400

  # 使用 SwinIR-L 大模型进行真实世界超分
  python run_predict.py --task real_sr --scale 4 --large_model \\
      --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \\
      --input photo.jpg

环境要求：
  pip install -r requirements.txt
  核心依赖：torch>=1.8.0, timm>=0.4.12, opencv-python, numpy, requests
"""

import argparse
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import requests
import torch

# SwinIR 模型架构（单文件，包含 SwinIR + Mlp + WindowAttention + SwinTransformerBlock
# + BasicLayer + RSTB + PatchEmbed + PatchUnEmbed + Upsample 等全部子模块）
from models.network_swinir import SwinIR as net
# PSNR / SSIM / PSNR-B 评估工具（来自 BasicSR 项目）
from utils import util_calculate_psnr_ssim as util


# ═══════════════════════════════════════════════════════════════════════════════════
# 1. 模型构建与权重加载
# ═══════════════════════════════════════════════════════════════════════════════════

def build_model(args):
    """
    根据 args.task 指定的任务类型，构建对应配置的 SwinIR 模型，并加载预训练权重。

    不同任务的模型配置（embed_dim 嵌入维度、depths 每个 stage 的 Transformer block 数量、
    num_heads 注意力头数、window_size 窗口大小、upsampler 上采样方式）严格对应论文中的设定：

      任务                upsampler          window_size  img_range  embed_dim   depths
      ─────────────────────────────────────────────────────────────────────────────────
      classical_sr       pixelshuffle       8            1.0        180         [6]*6
      lightweight_sr     pixelshuffledirect  8            1.0         60         [6]*4
      real_sr (M)        nearest+conv       8            1.0        180         [6]*6
      real_sr (L)        nearest+conv       8            1.0        240         [6]*9
      gray_dn            ""（空字符串）      8            1.0        180         [6]*6
      color_dn           ""（空字符串）      8            1.0        180         [6]*6
      jpeg_car           ""（空字符串）      7            255.0      180         [6]*6
      color_jpeg_car     ""（空字符串）      7            255.0      180         [6]*6

    预训练权重加载说明：
      - 大多数任务使用 checkpoint 中的 'params' 键加载 state_dict
      - real_sr 使用 'params_ema' 键（GAN 训练时的指数移动平均权重，EMA 平滑后推理效果更好）
      - 如果 checkpoint 中不存在 param_key_g 对应的键，则直接将整个 checkpoint 当作 state_dict
      - 使用 strict=True 确保所有权重都匹配，不匹配时会报错

    参数：
      args: argparse.Namespace，包含 task、scale、training_patch_size、large_model 等配置

    返回：
      已加载预训练权重并设置为 eval 模式的 SwinIR 模型实例（尚未移至 device）
    """

    # 001 经典图像超分辨率（bicubic 下采样）
    # 特点：使用 pixelshuffle 上采样（多级 Conv2d + PixelShuffle 级联），6 个 RSTB
    # training_patch_size 参数区分表 2 中两种训练配置：
    #   48 → DIV2K（800 张训练图）
    #   64 → DIV2K + Flickr2K（2650 张训练图）
    # 该参数仅影响 SwinIR 初始化时的 img_size 参数，不影响实际推理（图像不会被裁成 patch）
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size,
                    window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 轻量图像超分辨率
    # 特点：使用 pixelshuffledirect 上采样（仅单级 Conv2d + PixelShuffle，参数更少）
    # embed_dim 仅为 60（经典 SR 的 1/3），只有 4 个 RSTB
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 真实世界图像超分辨率
    # 特点：使用 nearest+conv 上采样（最近邻插值 + Conv 细化），相比 pixelshuffle 减少块效应
    # 分为中等模型 SwinIR-M 和大模型 SwinIR-L 两种
    elif args.task == 'real_sr':
        if not args.large_model:
            # SwinIR-M（中等）：6 个 RSTB，embed_dim=180，1conv 残差连接
            # 训练数据：DIV2K + Flickr2K + OST
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                        num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # SwinIR-L（大）：9 个 RSTB，embed_dim=240，3conv 残差连接（更深的网络但节省参数）
            # 训练数据：DIV2K + Flickr2K + OST + WED + FFHQ + Manga109 + SCUT-CTW1500
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                        embed_dim=240, num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        # real_sr 使用 GAN 训练，EMA 权重在去伪影方面表现更好
        param_key_g = 'params_ema'

    # 004 灰度图像去噪
    # 特点：单通道输入（in_chans=1），img_size=128（去噪需要较大感受野）
    # upsampler 为空字符串 → 不上采样，仅用 conv_last 输出
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 彩色图像去噪
    # 特点：三通道输入（in_chans=3），img_size=128，其余与灰度去噪一致
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 灰度 JPEG 压缩伪影去除
    # 特点：
    #   1. window_size=7 而非 8 —— 因为 JPEG 使用 8×8 块编码，window_size=7 能更好地对齐
    #      块边界，避免窗口跨越两个 JPEG 块导致注意力分散
    #   2. img_range=255.0 而非 1.0 —— 经验表明在 JPEG CAR 任务上更大的值域表现略好
    #   3. img_size=126（126 是 7 的整数倍，便于窗口划分）
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 彩色 JPEG 压缩伪影去除（三通道版本，其他与灰度版相同）
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    else:
        raise ValueError(f'不支持的任务类型: {args.task}')

    # 加载预训练权重
    # torch.load 自动处理 map_location（默认加载到 CPU）
    # load_state_dict 使用 strict=True 确保所有层都匹配
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(
        pretrained_model[param_key_g] if param_key_g in pretrained_model
        else pretrained_model,
        strict=True
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════════
# 2. 批量评估配置
# ═══════════════════════════════════════════════════════════════════════════════════

def get_batch_config(args):
    """
    根据任务类型确定批量评估的运行配置。

    此函数仅在批量文件夹评估模式（--folder_gt 或 --folder_lq）下调用，
    单张推理模式下直接使用 get_window_size 和用户指定的 --output。

    返回 (folder, save_dir, border, window_size)：
      folder       — 输入图像所在目录
                     • classical_sr / lightweight_sr：GT 目录（LQ 由 get_image_pair 按命名规则查找）
                     • real_sr：LQ 目录（无 GT）
                     • gray_dn / color_dn / jpeg_car / color_jpeg_car：GT 目录（LQ 在线退化生成）
      save_dir     — 结果保存目录，格式为 results/swinir_{task}_{参数}
                     例如：results/swinir_classical_sr_x4
                           results/swinir_gray_dn_noise25
                           results/swinir_jpeg_car_jpeg40
      border       — PSNR/SSIM 评估时需在图像四周裁剪的像素数
                     • SR 任务：border = scale（因为 bicubic 下采样在边界会有插值伪影）
                     • 去噪 / JPEG CAR / real_sr：border = 0（无需裁剪）
      window_size  — Swin Transformer 的注意力窗口大小
                     • JPEG CAR：7（对齐 8×8 JPEG 块边界）
                     • 其他任务：8

    返回的 folder 决定了批量模式下扫描哪个目录来获取输入图像列表。
    """

    # 经典 SR / 轻量 SR
    # border = scale：评估时裁剪 scale 像素的边界以消除 bicubic 下采样带来的边界伪影
    # 例如 scale=4 时，图像四周各裁剪 4 像素后再计算 PSNR/SSIM
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 真实世界 SR：无 GT，不需要裁剪边界
    # folder 指向 LQ 目录，直接推理并保存结果
    elif args.task == 'real_sr':
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'  # 大模型加上 _large 后缀区分
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 图像去噪（灰度和彩色）
    # border = 0：去噪任务不需要裁剪边界（噪声是全局的，不像 SR 有边界插值）
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8

    # JPEG 压缩伪影去除（灰度和彩色）
    # window_size=7（不是 8！）：因为 JPEG 编码使用 8×8 DCT 块，7 能更好地对齐块边界
    # border = 0
    elif args.task in ['jpeg_car', 'color_jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_gt
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size


# ═══════════════════════════════════════════════════════════════════════════════════
# 3. 图像加载（批量模式）
# ═══════════════════════════════════════════════════════════════════════════════════

def load_image_pair(args, path):
    """
    根据任务类型加载输入图像对（LQ + GT）。

    不同任务的加载策略因退化类型不同而有所区别：

    ┌─────────────────────┬──────────────────────────────────────────────────┐
    │ 任务               │ 加载策略                                         │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ classical_sr       │ GT 从 path（--folder_gt）读取 HR 图像            │
    │ lightweight_sr     │ LQ 从 --folder_lq 按命名规则查找（{name}x{scale}.ext）│
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ real_sr            │ GT = None（真实场景无 GT）                       │
    │                     │ LQ 从 path（--folder_lq）直接读取               │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ gray_dn / color_dn │ GT 从 path 读取，在线添加高斯噪声生成 LQ         │
    │                     │ 噪声种子固定为 seed=0，保证评估结果可完全复现   │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ jpeg_car           │ GT 从 path 读取，在线进行 JPEG 压缩生成 LQ      │
    │ color_jpeg_car     │ JPEG 压缩质量由 --jpeg 参数控制，seed 固定       │
    └─────────────────────┴──────────────────────────────────────────────────┘

    参数：
      args: 命令行参数（包含 task、scale、noise、jpeg、folder_lq）
      path: 当前处理的图像完整路径（来自 glob 遍历）

    返回 (imgname, img_lq, img_gt)：
      imgname  — 图像文件名（不含扩展名），用于保存结果时的命名
      img_lq   — 低质量输入图像，格式为 HWC-BGR, float32，值域 [0, 1]
                 灰度图为 H×W×1 三通道格式（第三维为 1）
      img_gt   — 真实高质量图像，格式同上，单张模式或 real_sr 时为 None
    """
    # 提取文件名和扩展名，如 path="/data/img_001.png" → ("img_001", ".png")
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # ── 001/002 经典 SR 和轻量 SR ───────────────────────────
    # LQ 文件名约定：{GT文件名}x{scale}{扩展名}
    # 例如 GT=img_001.png, scale=2 → LQ=img_001x2.png
    if args.task in ['classical_sr', 'lightweight_sr']:
        # 读取 GT 高分辨率原图（BGR, float32 [0,1]）
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # 按命名规则查找并读取对应的低分辨率图像
        img_lq = cv2.imread(
            f'{args.folder_lq}/{imgname}x{args.scale}{imgext}',
            cv2.IMREAD_COLOR
        ).astype(np.float32) / 255.

    # ── 003 真实世界 SR ────────────────────────────────────
    # 真实场景只有低质量输入，没有 GT
    elif args.task == 'real_sr':
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # ── 004 灰度去噪 ──────────────────────────────────────
    # 从 GT 在线生成含噪 LQ：LQ = GT + N(0, σ²)
    # σ = args.noise / 255（因为图像值域为 [0,1]）
    # np.random.seed(0) 确保每次运行生成的噪声完全相同，评估结果可复现
    elif args.task == 'gray_dn':
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        # 灰度图扩展为 H×W×1 三通道格式（保持与彩色图接口一致）
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # ── 005 彩色去噪 ──────────────────────────────────────
    # 与灰度去噪逻辑相同，但保留三通道彩色格式
    elif args.task == 'color_dn':
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)

    # ── 006 灰度 JPEG CAR ─────────────────────────────────
    # LQ 通过 GT → JPEG 压缩 → 解压 生成，模拟 JPEG 块效应和 ringing 伪影
    # 处理流程：
    #   1. 读取原图（可能是彩色图）
    #   2. 如果是彩色图 → 提取 Y 通道（BT.601 亮度分量，因为 JPEG 伪影主要在亮度通道上）
    #   3. 以指定质量因子 jpeg 进行 JPEG 编码 → 立即解码回图像
    #   4. 这样解码后的图像就会带有 JPEG 压缩伪影
    elif args.task == 'jpeg_car':
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # 如果读入的是彩色图（3 通道），转为 Y 通道（YCbCr 的亮度分量）
        # 因为 JPEG 伪影在色彩空间转换到亮度上更明显，且灰度任务只处理单通道
        if img_gt.ndim != 2:
            img_gt = util.bgr2ycbcr(img_gt, y_only=True)
        # cv2.imencode 进行 JPEG 压缩，IMWRITE_JPEG_QUALITY 控制质量（越低伪影越重）
        # cv2.imdecode 解压回来，此时图像已带有压缩伪影
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        # 扩展为 H×W×1 格式
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    # ── 006 彩色 JPEG CAR ─────────────────────────────────
    # 与灰度版类似，但保留三通道彩色格式
    elif args.task == 'color_jpeg_car':
        img_gt = cv2.imread(path)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 1)  # 1 表示解码为彩色图
        img_gt = img_gt.astype(np.float32) / 255.
        img_lq = img_lq.astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def get_window_size(task):
    """
    返回 Swin Transformer 的注意力窗口大小。

    JPEG 压缩伪影去除任务使用 window_size=7，其他任务使用 window_size=8。

    为什么 JPEG CAR 用 7？
      JPEG 算法将图像划分为互不重叠的 8×8 像素块，对每个块独立做 DCT 变换和量化。
      这意味着块边界处会产生明显的不连续性（blocking artifact）。如果 SwinIR 的
      注意力窗口也是 8×8，窗口边界可能与 JPEG 块边界重合，导致窗口注意力无法
      有效跨越块边界来"看到"并修复块效应。使用 7×7 窗口可以让窗口边界与 JPEG
      块边界形成错位，每个窗口都会横跨两个相邻 JPEG 块，从而更好地学习跨块修复。
    """
    return 7 if task in ('jpeg_car', 'color_jpeg_car') else 8


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 模型推理
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(img_lq, model, args, window_size, progress_callback=None, cancel_event=None):
    """
    模型推理核心函数，支持整图推理和分块（tile）推理两种模式。

    ┌─ 整图推理（args.tile is None）─────────────────────────────
    │ 直接将整个输入张量送入模型，简单直接。适合显存足够的情况。
    │
    └─ 分块推理（args.tile is not None）─────────────────────────
      用于处理大分辨率图像，避免 GPU 显存溢出（OOM）。

      分块推理流程：
        1. 计算分块：将 H×W 的图像按 tile 大小切分为有重叠的网格
           - 步长 stride = tile - tile_overlap，重叠区域确保拼接平滑
           - 最后一个块紧贴图像边缘（[h-tile, w-tile]），避免漏掉边缘像素
        2. 逐块推理：将每个图像块独立送入模型前向传播
        3. 加权融合：将各块输出累加到输出画布 E 上，同时每个像素位置记录
           被覆盖次数 W。最终 output = E / W（重叠区域被多次覆盖，取平均后平滑过渡）

      E 和 W 的含义：
        E — 累积输出画布，尺寸 (B, C, H*sf, W*sf)，每个像素值是所有覆盖该位置的
            块的模型输出之和
        W — 权重画布，尺寸同 E，每个像素值是该位置被 tile 覆盖的次数
            重叠区域覆盖次数多（因 stride < tile），边缘区域覆盖次数少
        E / W — 按覆盖次数做加权平均，重叠区域平滑过渡，消除 tile 之间的接缝

    参数：
      img_lq      — 输入低质量图像张量，形状 (N, C, H, W)，已经过窗口填充
      model       — 已加载权重并 evaled 的 SwinIR 模型
      args        — 命令行参数，关键字段：tile、tile_overlap、scale
                    tile=None → 整图推理；tile=400 → 用 400×400 的块分块推理
      window_size — 注意力窗口大小（7 或 8），tile 必须能被 window_size 整除

    返回：
      模型输出张量 (N, C, H*sf, W*sf)，其中 sf = args.scale
      在 forward 内部 SwinIR 已处理 mean 减法和 img_range 缩放
      调用方需要自行做裁剪填充、clamp 和颜色空间转换
    """

    # 未指定 tile 时，直接将整个图像送入模型（整图推理）
    if args.tile is None:
        return model(img_lq)

    # ── 分块推理 ─────────────────────────────────────────
    b, c, h, w = img_lq.size()

    # tile 不能超过图像本身尺寸（对于小图直接等于图像尺寸）
    tile = min(args.tile, h, w)
    # 断言：tile 必须是 window_size 的整数倍，否则 SwinIR 内部的 window_partition 会出错
    assert tile % window_size == 0, "tile 必须是 window_size 的倍数"

    tile_overlap = args.tile_overlap  # 相邻块之间的重叠像素数
    sf = args.scale                    # 超分倍数（去噪/JPEG CAR 时 sf=1）

    # 计算水平和垂直方向的滑动窗口起止索引
    # stride = tile - overlap，确保相邻块有 overlap 像素的重叠
    # 最后一个索引为 h-tile（或 w-tile），确保最后一块紧贴图像右/下边缘
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    # E: 累积输出像素值（求和），W: 累积每个像素被覆盖的次数（计数）
    # 输出尺寸 = 输入尺寸 × scale（SR 放大，去噪/JPEG CAR 保持）
    E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
    W = torch.zeros_like(E)

    # 进度计数器
    total_tiles = len(h_idx_list) * len(w_idx_list)
    tile_idx = 0

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # 打印进度（\r 回到行首，不换行刷新）
            tile_idx += 1
            print(f'\r分块推理: {tile_idx}/{total_tiles}', end='', flush=True)

            # 调用进度回调（供 Web 服务等外部调用者使用）
            if progress_callback:
                progress_callback(tile_idx, total_tiles)

            # 检查是否取消推理
            if cancel_event and cancel_event.is_set():
                print(f'\n推理已取消（{tile_idx}/{total_tiles}）')
                return None

            # 从大图中裁出当前块 [h_idx:h_idx+tile, w_idx:w_idx+tile]
            # 注意：此时大图已经过窗口填充，padding 区域由 flip padding 填充
            in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)

            # mask 全 1 张量，累加到 W 中表示对应位置的覆盖次数 +1
            # 重叠区域会在多次迭代中被多次累加，最终除以 W 实现平均
            out_patch_mask = torch.ones_like(out_patch)

            # 将块输出累加到 E 的对应位置（按 scale 缩放映射到输出空间）
            E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
            W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

    print()  # 分块结束后换行

    # 加权平均：E / W。重叠区域的像素值会取多次推理的平均值，实现平滑过渡
    return E.div_(W)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 后处理：张量 → 可保存图像
# ═══════════════════════════════════════════════════════════════════════════════

def tensor_to_image(output, task, h_old, w_old, scale):
    """
    将模型输出的 PyTorch 张量转换为可保存的 uint8 numpy 图像。

    完整转换流程：
      NCHW-RGB float32 [0,1] 张量
        → 裁剪填充区域 → 移除 batch 维度 → 移至 CPU → clamp [0,1]
        → numpy → 颜色空间转换（RGB→BGR）→ 缩放到 [0,255] → uint8

    参数：
      output — 模型输出张量，形状 (1, C, H_padded*scale, W_padded*scale)，值域大约 [0,1]
      task   — 任务类型，灰度图任务（gray_dn, jpeg_car）跳过颜色通道翻转
               （灰度图只有 1 个通道或 squeeze 后为 2D，无需 BGR↔RGB 转换）
      h_old  — 原始图像高度（填充前），用于裁剪掉填充区域
      w_old  — 原始图像宽度（填充前）
      scale  — 超分倍数，输出尺寸 = 原始尺寸 × scale

    返回：
      numpy 数组，格式为 HWC-BGR uint8（彩色）或 HW uint8（灰度）
      可直接通过 cv2.imwrite 保存为 PNG/JPEG 等格式
    """
    # 裁剪掉窗口填充部分，恢复到实际图像区域
    # 因为输入经过了反射填充（pad to window_size 的倍数），
    # 模型输出也会包含填充部分，需要裁剪回原始尺寸
    output = output[..., :h_old * scale, :w_old * scale]

    # 将张量从 GPU 移到 CPU，移除 batch 维度，clamp 到 [0,1]
    # clamp_ (in-place) 比 clamp 更高效，直接修改张量不复制
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # 彩色图：CHW-RGB → HWC-BGR (OpenCV 保存需要 BGR 顺序)
    # output[[2,1,0]] 表示取通道索引 2(R),1(G),0(B) 即 RGB→BGR
    # transpose(1,2,0) 将 CHW 转为 HWC
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

    # [0, 1] float32 → [0, 255] uint8（OpenCV 写入需要 uint8 格式）
    return (output * 255.0).round().astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    SwinIR 图像推理的主入口函数。

    根据是否提供 --input 参数自动切换运行模式：
      • 有 --input → 单张图像推理模式（不计算 PSNR/SSIM）
      • 无 --input → 批量文件夹评估模式（遍历 --folder_gt 目录，自动计算指标）

    执行流程概览：
      1. 解析命令行参数
      2. 选择计算设备（CUDA > MPS > CPU）
      3. 自动下载模型（如果本地不存在）
      4. 构建模型并加载权重
      5. [单张模式] 读取图像 → 窗口填充 → 推理 → 后处理 → 保存
      6. [批量模式] 遍历文件夹 → 加载图像对 → 推理 → 评估 PSNR/SSIM → 汇总输出
    """
    parser = argparse.ArgumentParser(description='SwinIR 图像推理 — 单张/批量，支持分块推理')

    # ── 任务参数 ──────────────────────────────────────────
    parser.add_argument('--task', type=str, default='real_sr',
                        choices=['classical_sr', 'lightweight_sr', 'real_sr',
                                 'gray_dn', 'color_dn', 'jpeg_car', 'color_jpeg_car'],
                        help='任务类型：classical_sr=经典SR, lightweight_sr=轻量SR, '
                             'real_sr=真实世界SR, gray_dn=灰度去噪, color_dn=彩色去噪, '
                             'jpeg_car=灰度JPEG去伪影, color_jpeg_car=彩色JPEG去伪影')
    parser.add_argument('--scale', type=int, default=1,
                        help='超分倍数：1/2/3/4/8（去噪和 JPEG CAR 任务请设为 1）')
    parser.add_argument('--noise', type=int, default=15,
                        help='高斯噪声标准差（仅 gray_dn 和 color_dn 有效）：15 / 25 / 50')
    parser.add_argument('--jpeg', type=int, default=40,
                        help='JPEG 压缩质量因子（仅 jpeg_car 和 color_jpeg_car 有效）：'
                             '10 / 20 / 30 / 40（值越小压缩率越高，伪影越严重）')
    parser.add_argument('--training_patch_size', type=int, default=128,
                        help='训练时使用的 patch 大小。仅用于 classical_sr 任务以区分两种'
                             '不同的训练配置（Table 2 in paper）：'
                             '48 = 仅 DIV2K 训练，64 = DIV2K+Flickr2K 训练。'
                             '注意：此参数不影响实际推理过程，图像不会按 patch 分割。')
    parser.add_argument('--large_model', action='store_true',
                        help='使用 SwinIR-L 大模型（仅 real_sr 有效），'
                             '9 个 RSTB，embed_dim=240。不加此参数默认使用 SwinIR-M。')

    # ── 模型路径 ──────────────────────────────────────────
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型文件路径（.pth 文件）。如果文件不存在，'
                             '脚本会自动从 GitHub Releases 下载到该路径。')

    # ── 输入源（二选一）───────────────────────────────────
    # --input：触发单张推理模式
    # --folder_gt / --folder_lq：触发批量文件夹评估模式
    parser.add_argument('--input', type=str, default=None,
                        help='单张输入图像路径。指定后将进入单张推理模式，'
                             '直接输出增强后的图像，不计算 PSNR/SSIM。')
    parser.add_argument('--folder_lq', type=str, default=None,
                        help='低质量（LQ）测试图像目录。用于批量模式中查找低分辨率输入。'
                             'classical_sr/lightweight_sr 需要此参数来定位 LQ 图像。')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='真实高质量（GT）测试图像目录。用于批量模式中读取 GT 图像'
                             '并计算 PSNR/SSIM。如果不提供（如 real_sr），则不计算指标。')

    # ── 分块推理参数 ──────────────────────────────────────
    parser.add_argument('--tile', type=int, default=None,
                        help='分块推理时的块大小（像素）。不指定则整图推理，适合小图。'
                             '大图推荐设为 400。注意：tile 必须是 window_size（7 或 8）'
                             '的整数倍，否则会报错。')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='分块推理时相邻块之间的重叠像素数（默认 32）。'
                             '重叠区域通过加权平均实现平滑过渡，消除块之间的拼接痕迹。')

    # ── 输出 ──────────────────────────────────────────────
    parser.add_argument('--output', type=str, default=None,
                        help='输出图像保存路径（仅单张推理模式有效）。'
                             '不指定则默认保存为：{输入文件名}_SwinIR.png')

    args = parser.parse_args()

    # ════════════════════════════════════════════════════════════════
    # 设备选择
    # ════════════════════════════════════════════════════════════════
    # 按优先级自动选择：CUDA（NVIDIA GPU）> MPS（Apple Silicon GPU）> CPU
    # MPS 是 PyTorch 对 Apple M1/M2/M3 芯片的 GPU 加速后端
    # 注意：MPS 后端的结果可能与 CUDA 有微小差异（浮点精度不同）
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')

    # ════════════════════════════════════════════════════════════════
    # 模型下载
    # ════════════════════════════════════════════════════════════════
    # 如果 model_path 指定的文件已在本地，直接加载。
    # 否则从 SwinIR 官方 GitHub Releases 自动下载。所有预训练模型约 100-300MB。
    if os.path.exists(args.model_path):
        print(f'加载模型: {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{os.path.basename(args.model_path)}'
        print(f'下载模型: {url}')
        r = requests.get(url, allow_redirects=True)
        with open(args.model_path, 'wb') as f:
            f.write(r.content)

    # ════════════════════════════════════════════════════════════════
    # 构建模型
    # ════════════════════════════════════════════════════════════════
    # build_model 内部完成：根据 task 选择配置 → 初始化 SwinIR → 加载权重
    model = build_model(args)
    model.eval()       # 切换到推理模式（关闭 Dropout、BatchNorm 等）
    model = model.to(device)  # 将模型参数移至目标设备

    # ── 判断运行模式 ─────────────────────────────────────
    # 有 --input 参数 → 单张模式；否则 → 批量模式
    is_single = args.input is not None

    if is_single:
        # ╔═══════════════════════════════════════════════════════════════╗
        # ║              单张图像推理模式                                  ║
        # ║  输入：一张低质量图像                                         ║
        # ║  输出：一张增强后的高质量图像                                 ║
        # ║  指标：不计算 PSNR/SSIM（没有 GT 参考图）                     ║
        # ╚═══════════════════════════════════════════════════════════════╝

        window_size = get_window_size(args.task)

        # 默认输出文件名：{原始文件名}_SwinIR.png
        if args.output is None:
            args.output = os.path.splitext(os.path.basename(args.input))[0] + '_SwinIR.png'

        # ── 读取输入图像 ──────────────────────────────────
        # OpenCV 默认读取为 BGR 格式，需要转换为 RGB 送入模型。
        # 灰度图任务（gray_dn、jpeg_car）直接读为单通道灰度。
        if args.task in ('gray_dn', 'jpeg_car'):
            # 灰度图：H×W → 扩展为 1×H×W（单通道 CHW）
            img_lq = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=0)
        else:
            # 彩色图：H×W×C(BGR) → C×H×W(RGB)
            # img[:,:, [2,1,0]] 将 BGR 通道重排为 RGB
            # transpose(2,0,1) 将 HWC 转为 CHW
            img_lq = cv2.imread(args.input, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))

        # 添加 batch 维度并移至设备：C×H×W → 1×C×H×W（NCHW）
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        # ── 推理 ──────────────────────────────────────────
        with torch.no_grad():  # 关闭梯度计算，节省显存和加速推理
            _, _, h_old, w_old = img_lq.size()

            # 窗口填充：将图像反射填充至 window_size 的整数倍
            # SwinIR 的 window_partition 要求 H 和 W 必须能被 window_size 整除
            # 使用 torch.flip 做反射填充（而非 zeros 填充），避免边界出现黑边/伪影
            # 填充区域在推理后会被裁剪掉（见 tensor_to_image）
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            # 对 H 维度做反射填充：将 img_lq 沿 dim=2 翻转后拼接到下方
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            # 对 W 维度做反射填充：将 img_lq 沿 dim=3 翻转后拼接到右侧
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            # 前向推理（内部根据 --tile 自动决定整图或分块推理）
            output = run_inference(img_lq, model, args, window_size)

        # ── 后处理并保存 ──────────────────────────────────
        result = tensor_to_image(output, args.task, h_old, w_old, args.scale)
        cv2.imwrite(args.output, result)
        print(f'结果已保存至 {args.output}')

    else:
        # ╔═══════════════════════════════════════════════════════════════╗
        # ║              批量文件夹评估模式                                ║
        # ║  输入：文件夹内所有图像                                       ║
        # ║  输出：增强后的图像 + PSNR/SSIM 指标（如有 GT）               ║
        # ║  保存：results/swinir_{task}_{params}/                        ║
        # ╚═══════════════════════════════════════════════════════════════╝

        # 获取批量评估配置：folder（扫描哪个目录）、save_dir（结果存哪里）、
        # border（PSNR 裁剪边界）、window_size（注意力窗口大小）
        folder, save_dir, border, window_size = get_batch_config(args)
        os.makedirs(save_dir, exist_ok=True)

        # 初始化指标收集器
        # 使用 OrderedDict 保证输出顺序。分别记录 RGB 和 Y 通道（亮度）的 PSNR/SSIM，
        # 以及 JPEG CAR 专用的 PSNR-B（考虑了块效应的感知质量指标）
        test_results = OrderedDict()
        test_results['psnr'] = []      # PSNR (RGB)
        test_results['ssim'] = []      # SSIM (RGB)
        test_results['psnr_y'] = []    # PSNR on Y channel (luminance)
        test_results['ssim_y'] = []    # SSIM on Y channel
        test_results['psnrb'] = []     # PSNR-B（仅 JPEG CAR 任务）
        test_results['psnrb_y'] = []   # PSNR-B on Y channel（仅彩色 JPEG CAR）
        psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

        # 遍历文件夹内所有图像文件（按文件名排序，确保结果可复现）
        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            # ── 加载图像对 ──────────────────────────────────
            # load_image_pair 根据任务类型自动处理：SR 读文件对，去噪在线加噪，JPEG 在线压缩
            imgname, img_lq, img_gt = load_image_pair(args, path)

            # ── 预处理：HWC-BGR → CHW-RGB → NCHW → device
            # 灰度图（shape[2]==1）不需要颜色通道转换
            img_lq = np.transpose(
                img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                (2, 0, 1)
            )
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

            # ── 推理（与单张模式相同流程）─────────────────────
            with torch.no_grad():
                _, _, h_old, w_old = img_lq.size()
                # 反射填充至 window_size 的整数倍
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = run_inference(img_lq, model, args, window_size)

            # ── 后处理并保存 ─────────────────────────────────
            result = tensor_to_image(output, args.task, h_old, w_old, args.scale)
            cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', result)

            # ── PSNR/SSIM 评估（仅当有 GT 参考图时）───────
            if img_gt is not None:
                # GT 图像也需转 uint8 用于评估
                img_gt = (img_gt * 255.0).round().astype(np.uint8)
                # 裁剪 GT 到与输出相同的区域（去掉 padding 对应的区域）
                img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]
                img_gt = np.squeeze(img_gt)

                # 计算 PSNR 和 SSIM（RGB 通道）
                # crop_border=border 用于裁剪图像边界像素后再计算
                # 对于 SR 任务 border=scale，避免 bicubic 边界效应影响指标
                psnr = util.calculate_psnr(result, img_gt, crop_border=border)
                ssim = util.calculate_ssim(result, img_gt, crop_border=border)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                # 如果是三通道彩色图，额外计算 Y 通道（亮度）的 PSNR/SSIM
                # 亮度通道指标更接近人眼感知，也是论文中常用的评估标准
                if img_gt.ndim == 3:
                    psnr_y = util.calculate_psnr(result, img_gt, crop_border=border, test_y_channel=True)
                    ssim_y = util.calculate_ssim(result, img_gt, crop_border=border, test_y_channel=True)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)

                # JPEG CAR 任务额外计算 PSNR-B（考虑了块效应的感知质量指标）
                # PSNR-B 对 blocking artifact 有额外的惩罚项
                if args.task in ('jpeg_car', 'color_jpeg_car'):
                    psnrb = util.calculate_psnrb(result, img_gt, crop_border=border, test_y_channel=False)
                    test_results['psnrb'].append(psnrb)
                    if args.task == 'color_jpeg_car':
                        psnrb_y = util.calculate_psnrb(result, img_gt, crop_border=border, test_y_channel=True)
                        test_results['psnrb_y'].append(psnrb_y)

                # 逐张打印指标（便于跟踪每张图像的恢复质量）
                print(f'测试 {idx:2d}  {imgname:20s} - '
                      f'PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}; PSNRB: {psnrb:.2f} dB; '
                      f'PSNR_Y: {psnr_y:.2f} dB; SSIM_Y: {ssim_y:.4f}; PSNRB_Y: {psnrb_y:.2f} dB.')
            else:
                # real_sr 任务无 GT，只打印处理进度
                print(f'测试 {idx:2d}  {imgname:20s}')

        # ── 汇总输出平均指标 ────────────────────────────────
        if img_gt is not None:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            print(f'\n{save_dir}')
            print(f'-- 平均 PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')

            if test_results['psnr_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                print(f'-- 平均 PSNR_Y/SSIM_Y (亮度): {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}')

            if test_results['psnrb']:
                ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
                print(f'-- 平均 PSNRB: {ave_psnrb:.2f} dB')

            if test_results['psnrb_y']:
                ave_psnrb_y = sum(test_results['psnrb_y']) / len(test_results['psnrb_y'])
                print(f'-- 平均 PSNRB_Y (亮度): {ave_psnrb_y:.2f} dB')


if __name__ == '__main__':
    main()
