#!/usr/bin/env python
"""
SwinIR 单张图像推理脚本
从 predict.py 中提取的核心推理逻辑，去除 Cog 依赖，可直接命令行调用。
支持任务：real_sr（真实世界超分）、gray_dn（灰度去噪）、color_dn（彩色去噪）、jpeg_car（JPEG去伪影）
"""
import argparse
import os
import cv2
import numpy as np
import torch
from main_test_swinir import define_model, setup

def main():
    parser = argparse.ArgumentParser(description='SwinIR 单张图像推理')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径（默认为 输入文件名_SwinIR.png）')
    parser.add_argument('--task', type=str, default='real_sr',
                        choices=['real_sr', 'gray_dn', 'color_dn', 'jpeg_car'],
                        help='任务类型：real_sr=真实世界超分, gray_dn=灰度去噪, color_dn=彩色去噪, jpeg_car=JPEG去伪影')
    parser.add_argument('--scale', type=int, default=4, help='超分倍数（仅 real_sr 适用）')
    parser.add_argument('--noise', type=int, default=15, help='噪声等级（仅 gray_dn/color_dn 适用）：15/25/50')
    parser.add_argument('--jpeg', type=int, default=40, help='JPEG 质量因子（仅 jpeg_car 适用）：10/20/30/40')
    parser.add_argument('--large_model', action='store_true', help='使用大模型（仅 real_sr 适用）')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    args = parser.parse_args()

    # 设备选择：优先 CUDA > MPS（Apple Silicon）> CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')

    # 这些参数在 define_model 和 setup 中会被用到，设为 None 不给路径
    args.folder_gt = None
    args.folder_lq = None

    # 构建模型（根据 task 等参数初始化对应架构）
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup 返回 folder（图像目录）、save_dir（结果保存目录）、border（剪裁边界）、window_size（窗口大小）
    folder, save_dir, border, window_size = setup(args)
    if args.output is None:
        # 默认输出文件名：原始文件名_SwinIR.png
        args.output = os.path.splitext(os.path.basename(args.input))[0] + '_SwinIR.png'

    # 读取输入图像：BGR -> float32 [0, 1]
    img_lq = cv2.imread(args.input, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    # HWC-BGR -> CHW-RGB
    img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))
    # CHW-RGB -> NCHW-RGB（添加 batch 维度并移至设备）
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, h_old, w_old = img_lq.size()
        # 将输入填充为 window_size 的整数倍（使用镜像翻转填充）
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        # 前向推理
        output = model(img_lq)
        # 裁剪掉填充部分，恢复到原始尺寸 × scale
        output = output[..., :h_old * args.scale, :w_old * args.scale]

    # 后处理：Tensor -> numpy -> 裁切值域 -> 转换颜色空间 -> uint8
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        # CHW-RGB -> HWC-BGR
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    # [0, 1] -> [0, 255] uint8
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(args.output, output)
    print(f'结果已保存至 {args.output}')


if __name__ == '__main__':
    main()
