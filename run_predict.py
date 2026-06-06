#!/usr/bin/env python
"""
SwinIR 图像推理脚本（完全独立，整合 main_test_swinir.py 全部功能）

支持单张图像推理和批量文件夹评估两种模式，去除 Cog 依赖，可直接命令行调用。

来源：从 main_test_swinir.py 提取 define_model、setup、get_image_pair、test 等全部核心逻辑。

模型架构说明（models/network_swinir.py 中的 SwinIR 类）：
  浅层特征提取（一个 3×3 卷积）
    → 深层特征提取（多个 RSTB 残差 Swin Transformer 块）
    → 图像重建（upsampler 决定重建方式）

不同任务通过 upsampler 参数区分重建方式：
  - pixelshuffle     经典 SR（conv + PixelShuffle 链）
  - pixelshuffledirect 轻量 SR（单 conv + PixelShuffle，参数更少）
  - nearest+conv     真实世界 SR（最近邻上采样 + conv，减少块效应）
  - 空字符串        去噪/JPEG CAR（不上采样，仅 conv_last）

支持的全部任务：
  - classical_sr   经典 bicubic 图像超分辨率
  - lightweight_sr 轻量图像超分辨率
  - real_sr        真实世界图像超分辨率
  - gray_dn        灰度图像去噪
  - color_dn       彩色图像去噪
  - jpeg_car       灰度 JPEG 压缩伪影去除
  - color_jpeg_car 彩色 JPEG 压缩伪影去除

用法示例：
  # 单张图像推理（仅输出增强后的图像）
  python run_predict.py --task real_sr --scale 4 --model_path <模型路径> --input <输入图>

  # 批量文件夹评估（输出图像 + PSNR/SSIM 指标）
  python run_predict.py --task classical_sr --scale 2 --training_patch_size 48 \
      --model_path <模型路径> --folder_gt <GT目录> --folder_lq <LQ目录>

  # 大图分块推理（避免显存不足）
  python run_predict.py --task real_sr --scale 4 --model_path <模型路径> --input <大图> --tile 400
"""

import argparse
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import requests
import torch

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util


# ═══════════════════════════════════════════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════════════════════════════════════════

def define_model(args):
    """
    根据任务类型构建 SwinIR 模型并加载预训练权重。

    每个任务的模型配置（embed_dim、depths、num_heads、window_size 等）
    与论文 Table 1-6 中的设定一一对应。

    权重加载说明：
      - 大多数任务使用 checkpoint 中的 'params' 键
      - real_sr 使用 'params_ema'（GAN 训练时的指数移动平均权重，推理效果更好）
      - 如果 checkpoint 中没有这些键，则直接作为 state_dict 加载

    返回构建好的模型实例。
    """

    # 001 经典图像超分辨率（bicubic 下采样）
    # 使用 pixelshuffle 上采样，training_patch_size 区分表 2 中两种训练配置
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size,
                    window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 轻量图像超分辨率
    # pixelshuffledirect 上采样比 pixelshuffle 参数更少，4 个 RSTB
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 真实世界图像超分辨率
    # nearest+conv 上采样可减少块效应
    elif args.task == 'real_sr':
        if not args.large_model:
            # SwinIR-M：6 个 RSTB，embed_dim=180
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                        num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # SwinIR-L：9 个 RSTB，embed_dim=240，3conv 残差连接节省参数
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                        embed_dim=240, num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 灰度图像去噪（单通道输入，upsampler 为空表示不上采样）
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 彩色图像去噪（三通道输入）
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 灰度 JPEG 压缩伪影去除
    # window_size=7 是因为 JPEG 使用 8×8 块编码，7 能更好地对齐块边界
    # img_range=255 在此任务上比 1.0 略好
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 彩色 JPEG 压缩伪影去除
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    else:
        raise ValueError(f'不支持的任务类型: {args.task}')

    # 加载预训练权重：先尝试 param_key_g 键（如 'params' 或 'params_ema'），
    # 如果没有则直接把整个 checkpoint 当作 state_dict 使用
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(
        pretrained_model[param_key_g] if param_key_g in pretrained_model
        else pretrained_model,
        strict=True
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 配置与图像加载
# ═══════════════════════════════════════════════════════════════════════════════

def setup(args):
    """
    根据任务类型确定运行配置。

    返回 (folder, save_dir, border, window_size)：
      folder:      输入图像所在目录（对于有 GT 的任务是 GT 目录，real_sr 是 LQ 目录）
      save_dir:    结果保存目录
      border:      PSNR/SSIM 评估时需裁剪的边界像素数（SR 任务按 scale 裁剪边界避免边界效应）
      window_size: Swin Transformer 的窗口大小
    """

    # 经典 SR / 轻量 SR：评估 PSNR 时需要裁剪 scale 像素的边界
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 真实世界 SR：无 GT，不需要裁剪
    elif args.task == 'real_sr':
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 图像去噪
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8

    # JPEG 压缩伪影去除：window_size=7
    elif args.task in ['jpeg_car', 'color_jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_gt
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    """
    根据任务类型加载输入图像对（LQ + GT）。

    不同任务的加载策略：
      - classical_sr / lightweight_sr：从 --folder_lq 加载对应的低分辨率图像，GT 为 path 指向的高分辨率图像
      - real_sr：直接从 --folder_lq 读取低质量图像，无 GT
      - gray_dn / color_dn：读取 GT 图像后在线添加高斯噪声生成 LQ（固定 seed=0 保证可复现）
      - jpeg_car / color_jpeg_car：读取 GT 图像后在线进行 JPEG 压缩生成 LQ
      - 单张图像模式（--input）：直接读取输入图像作为 LQ，无 GT

    返回 (imgname, img_lq, img_gt)：
      imgname: 图像文件名（不含扩展名）
      img_lq:   低质量图像 (HWC-BGR, float32 [0,1])
      img_gt:   真实高质量图像，单张模式/real_sr 时为 None
    """
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001/002 经典 SR / 轻量 SR：根据 GT 文件名查找对应的 LQ 图像
    # LQ 文件命名规则：{imgname}x{scale}{imgext}，如 img_001x2.png
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = cv2.imread(
            f'{args.folder_lq}/{imgname}x{args.scale}{imgext}',
            cv2.IMREAD_COLOR
        ).astype(np.float32) / 255.

    # 003 真实世界 SR：仅加载 LQ 图像
    elif args.task == 'real_sr':
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # 004 灰度去噪：加载灰度 GT，在线添加高斯噪声生成 LQ
    elif args.task == 'gray_dn':
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        np.random.seed(seed=0)  # 固定种子保证噪声可复现
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # 005 彩色去噪：加载彩色 GT，在线添加高斯噪声生成 LQ
    elif args.task == 'color_dn':
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)

    # 006 灰度 JPEG CAR：加载灰度 GT，在线进行 JPEG 压缩生成 LQ
    elif args.task == 'jpeg_car':
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # 如果读入的是彩色图，转为 Y 通道（YCbCr 的亮度分量）
        if img_gt.ndim != 2:
            img_gt = util.bgr2ycbcr(img_gt, y_only=True)
        # JPEG 压缩 → 解压 模拟 JPEG 伪影
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    # 006 彩色 JPEG CAR：加载彩色 GT，在线进行 JPEG 压缩生成 LQ
    elif args.task == 'color_jpeg_car':
        img_gt = cv2.imread(path)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 1)
        img_gt = img_gt.astype(np.float32) / 255.
        img_lq = img_lq.astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def get_window_size(task):
    """JPEG CAR 用 window_size=7（对齐 JPEG 8×8 块边界），其余任务用 8"""
    return 7 if task in ('jpeg_car', 'color_jpeg_car') else 8


# ═══════════════════════════════════════════════════════════════════════════════
# 推理
# ═══════════════════════════════════════════════════════════════════════════════

def test(img_lq, model, args, window_size):
    """
    模型推理入口，支持整图推理和分块（tile）推理两种模式。

    分块推理原理（用于大图避免显存不足）：
      1. 将输入图像按 tile 大小切分成若干重叠的块
      2. 逐块送入模型推理
      3. 将各块输出按加权平均拼回完整图像（重叠区域被多次累加后除以计数，实现平滑过渡）

    参数：
      img_lq:      输入低质量图像张量 (NCHW)
      model:       已加载权重的 SwinIR 模型
      args:        命令行参数（包含 tile、tile_overlap、scale）
      window_size: 窗口大小（tile 需为其整数倍）

    返回模型输出张量。
    """

    if args.tile is None:
        return model(img_lq)

    b, c, h, w = img_lq.size()
    tile = min(args.tile, h, w)
    assert tile % window_size == 0, "tile 必须是 window_size 的倍数"

    tile_overlap = args.tile_overlap
    sf = args.scale

    # 计算滑动步长和行列索引列表（最后一个块紧贴边缘）
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    # E 累积输出像素值，W 累积每个像素被覆盖的次数（用于加权平均）
    E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)
            E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
            W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

    return E.div_(W)


# ═══════════════════════════════════════════════════════════════════════════════
# 后处理
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess(output, task, h_old, w_old, scale):
    """
    将模型输出张量转换为可保存的 uint8 numpy 图像。

    Tensor(NCHW RGB [0,1]) → numpy(HWC BGR [0,255] uint8)

    参数：
      output: 模型输出张量
      task:   任务类型（灰度图特殊处理）
      h_old:  原始图像高度（用于裁剪填充）
      w_old:  原始图像宽度
      scale:  超分倍数

    返回 (HWC BGR uint8) numpy 数组。
    """
    # 裁剪掉填充部分
    output = output[..., :h_old * scale, :w_old * scale]
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # 彩色图：CHW-RGB → HWC-BGR
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

    return (output * 255.0).round().astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SwinIR 图像推理')

    # 任务参数
    parser.add_argument('--task', type=str, default='real_sr',
                        choices=['classical_sr', 'lightweight_sr', 'real_sr',
                                 'gray_dn', 'color_dn', 'jpeg_car', 'color_jpeg_car'],
                        help='任务类型')
    parser.add_argument('--scale', type=int, default=1,
                        help='超分倍数：1/2/3/4/8（去噪和 JPEG CAR 任务设 1）')
    parser.add_argument('--noise', type=int, default=15,
                        help='噪声等级（仅 gray_dn/color_dn）：15/25/50')
    parser.add_argument('--jpeg', type=int, default=40,
                        help='JPEG 质量因子（仅 jpeg_car/color_jpeg_car）：10/20/30/40')
    parser.add_argument('--training_patch_size', type=int, default=128,
                        help='训练时的 patch 大小，仅用于区分 classical_sr 的两种配置 '
                             '（Table 2：DIV2K 用 48，DIV2K+Flickr2K 用 64）。'
                             '不影响实际推理，图像不会按 patch 分割测试。')
    parser.add_argument('--large_model', action='store_true',
                        help='使用 SwinIR-L 大模型（仅 real_sr 适用）')

    # 模型路径
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径（.pth），如不存在则自动从 GitHub Releases 下载')

    # 输入源：二选一
    #   --input：单张图像推理
    #   --folder_gt / --folder_lq：批量文件夹评估
    parser.add_argument('--input', type=str, default=None,
                        help='单张输入图像路径')
    parser.add_argument('--folder_lq', type=str, default=None,
                        help='低质量测试图像目录（批量模式）')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='真实高质量图像目录（批量模式，用于 PSNR/SSIM 评估）')

    # 分块推理
    parser.add_argument('--tile', type=int, default=None,
                        help='分块大小，大图推理避免显存不足（需为 window_size 的倍数，建议 400）')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='分块之间的重叠像素数（默认 32）')

    # 输出
    parser.add_argument('--output', type=str, default=None,
                        help='输出图像路径（单张模式，默认为 输入文件名_SwinIR.png）')

    args = parser.parse_args()

    # ── 设备选择 ──────────────────────────────────────────
    # 优先级：CUDA > MPS（Apple Silicon）> CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')

    # ── 自动下载模型 ──────────────────────────────────────
    if os.path.exists(args.model_path):
        print(f'加载模型: {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{os.path.basename(args.model_path)}'
        print(f'下载模型: {url}')
        r = requests.get(url, allow_redirects=True)
        with open(args.model_path, 'wb') as f:
            f.write(r.content)

    # ── 构建模型 ──────────────────────────────────────────
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # ── 判断模式：单张推理 vs 批量文件夹评估 ──────────────
    is_single = args.input is not None

    if is_single:
        # ================================================================
        # 单张图像推理模式
        # 直接读取输入图像作为 LQ，不计算 PSNR/SSIM
        # ================================================================

        window_size = get_window_size(args.task)

        if args.output is None:
            args.output = os.path.splitext(os.path.basename(args.input))[0] + '_SwinIR.png'

        # 读取图像
        if args.task in ('gray_dn', 'jpeg_car'):
            img_lq = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=0)  # H×W → 1×H×W
        else:
            img_lq = cv2.imread(args.input, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR → CHW-RGB

        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # → NCHW

        # 推理
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            # 反射填充至 window_size 的整数倍
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)

        # 后处理并保存
        result = postprocess(output, args.task, h_old, w_old, args.scale)
        cv2.imwrite(args.output, result)
        print(f'结果已保存至 {args.output}')

    else:
        # ================================================================
        # 批量文件夹评估模式
        # 遍历文件夹，逐张推理并计算 PSNR/SSIM（如有 GT）
        # ================================================================

        folder, save_dir, border, window_size = setup(args)
        os.makedirs(save_dir, exist_ok=True)

        # 初始化评估指标记录
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []
        test_results['psnrb'] = []
        test_results['psnrb_y'] = []
        psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            # 加载图像对
            imgname, img_lq, img_gt = get_image_pair(args, path)

            # 预处理：HWC-BGR → CHW-RGB → NCHW → device
            img_lq = np.transpose(
                img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                (2, 0, 1)
            )
            img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = test(img_lq, model, args, window_size)

            # 后处理保存
            result = postprocess(output, args.task, h_old, w_old, args.scale)
            cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', result)

            # PSNR/SSIM 评估
            if img_gt is not None:
                img_gt = (img_gt * 255.0).round().astype(np.uint8)
                img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]
                img_gt = np.squeeze(img_gt)

                psnr = util.calculate_psnr(result, img_gt, crop_border=border)
                ssim = util.calculate_ssim(result, img_gt, crop_border=border)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if img_gt.ndim == 3:
                    psnr_y = util.calculate_psnr(result, img_gt, crop_border=border, test_y_channel=True)
                    ssim_y = util.calculate_ssim(result, img_gt, crop_border=border, test_y_channel=True)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)

                if args.task in ('jpeg_car', 'color_jpeg_car'):
                    psnrb = util.calculate_psnrb(result, img_gt, crop_border=border, test_y_channel=False)
                    test_results['psnrb'].append(psnrb)
                    if args.task == 'color_jpeg_car':
                        psnrb_y = util.calculate_psnrb(result, img_gt, crop_border=border, test_y_channel=True)
                        test_results['psnrb_y'].append(psnrb_y)

                print(f'测试 {idx:2d}  {imgname:20s} - '
                      f'PSNR: {psnr:.2f} dB; SSIM: {ssim:.4f}; PSNRB: {psnrb:.2f} dB; '
                      f'PSNR_Y: {psnr_y:.2f} dB; SSIM_Y: {ssim_y:.4f}; PSNRB_Y: {psnrb_y:.2f} dB.')
            else:
                print(f'测试 {idx:2d}  {imgname:20s}')

        # 汇总指标
        if img_gt is not None:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            print(f'\n{save_dir}')
            print(f'-- 平均 PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')
            if test_results['psnr_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                print(f'-- 平均 PSNR_Y/SSIM_Y: {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}')
            if test_results['psnrb']:
                ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
                print(f'-- 平均 PSNRB: {ave_psnrb:.2f} dB')
            if test_results['psnrb_y']:
                ave_psnrb_y = sum(test_results['psnrb_y']) / len(test_results['psnrb_y'])
                print(f'-- 平均 PSNRB_Y: {ave_psnrb_y:.2f} dB')


if __name__ == '__main__':
    main()
