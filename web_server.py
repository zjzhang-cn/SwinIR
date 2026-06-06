#!/usr/bin/env python
"""
SwinIR Web 推理服务
===================

基于 FastAPI 将 run_predict.py 的推理功能封装为 REST API，支持通过 HTTP 接口上传图像进行增强。

启动服务：
  python web_server.py --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
  python web_server.py --task real_sr --scale 4 --model_path <路径> --port 8080

API 端点：
  POST /predict   — 上传图像进行增强（multipart/form-data）
  GET  /tasks     — 获取支持的任务列表及参数说明
  GET  /health    — 服务健康检查
  GET  /docs      — FastAPI 自动生成的交互式 API 文档（Swagger UI）
"""

import argparse
import io
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

# 导入 run_predict.py 中的核心函数
from run_predict import (
    build_model,
    get_window_size,
    run_inference,
    tensor_to_image,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 全局模型实例（服务启动时加载，所有请求复用）
# ═══════════════════════════════════════════════════════════════════════════════

# 使用 uvicorn 的多 worker 模式时每个 worker 独立加载
model = None
device = None
base_args = None  # 基础配置（model_path、task、scale 等）


def create_args(**overrides):
    """
    创建一个简易的配置对象（类似 argparse.Namespace），供 build_model 和
    run_inference 使用。base_args 中的默认值可被 overrides 覆盖。
    """
    defaults = {
        'task': 'real_sr',
        'scale': 4,
        'noise': 15,
        'jpeg': 40,
        'training_patch_size': 128,
        'large_model': False,
        'model_path': '',
        'tile': None,
        'tile_overlap': 32,
    }
    if base_args:
        defaults.update({k: v for k, v in vars(base_args).items()})
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def load_model(model_path: str):
    """
    加载 SwinIR 模型并移至合适的计算设备。
    返回 (model, device)。
    """
    # 设备优先级：CUDA > MPS（Apple Silicon）> CPU
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    print(f'[初始化] 计算设备: {dev}')

    # 构建模型配置
    args = create_args(model_path=model_path)
    m = build_model(args)
    m.eval()
    m = m.to(dev)
    print(f'[初始化] 模型加载完成: {model_path}')
    return m, dev


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI 应用
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title='SwinIR 图像增强服务',
    description='基于 Swin Transformer 的图像恢复 REST API。支持超分辨率、去噪、JPEG 去伪影等 7 种任务。',
    version='1.0.0',
)

# 允许跨域（前端 HTML 页面可从任意域名调用 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# 启动事件：加载模型
@app.on_event('startup')
async def startup():
    global model, device, base_args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--task', type=str, default='real_sr')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--noise', type=int, default=15)
    parser.add_argument('--jpeg', type=int, default=40)
    parser.add_argument('--large_model', action='store_true')
    parser.add_argument('--tile', type=int, default=None)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')

    # 从命令行解析（兼容 python web_server.py --xxx 的调用方式）
    # 过滤掉 uvicorn 不识别的参数
    known, _ = parser.parse_known_args()
    base_args = known
    model, device = load_model(known.model_path)


# ═══════════════════════════════════════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════════════════════════════════════

@app.get('/')
async def index():
    """返回交互式 Web 页面。"""
    ui_path = os.path.join(os.path.dirname(__file__), 'web_ui.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type='text/html')
    return HTMLResponse('<h1>web_ui.html 未找到</h1>', status_code=404)


@app.get('/health')
async def health():
    """服务健康检查 — 返回模型和设备状态。"""
    return {
        'status': 'ok' if model is not None else 'model_not_loaded',
        'device': str(device) if device else 'none',
        'task': base_args.task if base_args else 'unknown',
    }


@app.get('/tasks')
async def list_tasks():
    """获取所有支持的任务类型及参数说明。"""
    return {
        'tasks': [
            {
                'key': 'classical_sr',
                'name': '经典图像超分辨率',
                'description': '对 bicubic 下采样的低分辨率图像进行超分辨率重建',
                'scale': '2 / 3 / 4 / 8',
                'extra_params': ['training_patch_size'],
                'note': 'training_patch_size=48 表示 DIV2K 训练，64 表示 DIV2K+Flickr2K 训练',
            },
            {
                'key': 'lightweight_sr',
                'name': '轻量图像超分辨率',
                'description': '参数更少的快速超分辨率模型',
                'scale': '2 / 3 / 4',
                'extra_params': [],
                'note': '相比 classical_sr 参数量更少，速度快',
            },
            {
                'key': 'real_sr',
                'name': '真实世界图像超分辨率',
                'description': '对真实世界中未知退化的低质量图像进行超分辨率重建（默认任务）',
                'scale': '4（固定）',
                'extra_params': ['large_model'],
                'note': '使用 nearest+conv 上采样减少块效应；可加 --large_model 使用 SwinIR-L',
            },
            {
                'key': 'gray_dn',
                'name': '灰度图像去噪',
                'description': '去除灰度图像中的高斯噪声',
                'scale': '1（不去噪时不放大）',
                'extra_params': ['noise'],
                'note': 'noise 可选 15 / 25 / 50，对应噪声标准差 σ',
            },
            {
                'key': 'color_dn',
                'name': '彩色图像去噪',
                'description': '去除彩色图像中的高斯噪声',
                'scale': '1',
                'extra_params': ['noise'],
                'note': 'noise 可选 15 / 25 / 50',
            },
            {
                'key': 'jpeg_car',
                'name': '灰度 JPEG 压缩伪影去除',
                'description': '修复灰度图像中因 JPEG 压缩引起的块效应和 ringing 伪影',
                'scale': '1',
                'extra_params': ['jpeg'],
                'note': 'jpeg 可选 10 / 20 / 30 / 40，值越小压缩率越高伪影越严重',
            },
            {
                'key': 'color_jpeg_car',
                'name': '彩色 JPEG 压缩伪影去除',
                'description': '修复彩色图像中因 JPEG 压缩引起的块效应和 ringing 伪影',
                'scale': '1',
                'extra_params': ['jpeg'],
                'note': 'jpeg 可选 10 / 20 / 30 / 40',
            },
        ],
        'common_params': {
            'tile': '分块大小（像素），不传则整图推理，大图建议 400',
            'tile_overlap': '分块重叠像素数（默认 32），值越大拼接越平滑但越慢',
        },
        'supported_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    }


@app.post('/predict')
async def predict(
    image: UploadFile = File(..., description='待增强的输入图像（支持 jpg/png/bmp/tiff）'),
    task: str = Form('real_sr', description='任务类型，默认 real_sr'),
    scale: int = Form(4, ge=1, le=8, description='超分倍数 (1/2/3/4/8)'),
    noise: int = Form(15, description='噪声等级 (15/25/50)，仅去噪任务有效'),
    jpeg:  int = Form(40, description='JPEG 质量因子 (10/20/30/40)，仅 JPEG CAR 有效'),
    large_model: bool = Form(False, description='是否使用 SwinIR-L 大模型，仅 real_sr 有效'),
    tile: int = Form(None, description='分块大小，大图推理避免显存不足（需为 window_size 的倍数）'),
    tile_overlap: int = Form(32, description='分块重叠像素数'),
):
    """
    上传一张图像进行增强处理，返回增强后的图像。

    支持的任务类型：
      - real_sr：真实世界超分辨率（默认）
      - gray_dn / color_dn：图像去噪
      - jpeg_car / color_jpeg_car：JPEG 压缩伪影去除

    返回：
      - 成功：返回增强后的 PNG 图像（image/png）
      - 失败：返回 JSON 错误信息
    """

    if model is None:
        raise HTTPException(status_code=503, detail='模型尚未加载，请稍后重试')

    # 验证任务类型
    valid_tasks = ['classical_sr', 'lightweight_sr', 'real_sr',
                   'gray_dn', 'color_dn', 'jpeg_car', 'color_jpeg_car']
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f'不支持的任务类型: {task}')

    # ── 读取上传的图像 ──────────────────────────────────
    try:
        contents = await image.read()
        # 根据任务类型选择读取方式
        if task in ('gray_dn', 'jpeg_car'):
            # 灰度任务：通过 PIL 读取为灰度图，再转为 OpenCV 格式
            pil_img = Image.open(io.BytesIO(contents)).convert('L')
            img_lq = np.array(pil_img).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=0)  # H×W → 1×H×W
        else:
            # 彩色任务：numpy → OpenCV BGR → float32 [0,1]
            npy = np.frombuffer(contents, np.uint8)
            img_lq = cv2.imdecode(npy, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            if img_lq is None:
                raise ValueError('无法解码图像')
            # HWC-BGR → CHW-RGB
            img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'图像读取失败: {str(e)}')

    # ── 推理配置 ────────────────────────────────────────
    args = create_args(
        task=task,
        scale=scale,
        noise=noise,
        jpeg=jpeg,
        large_model=large_model,
        tile=tile,
        tile_overlap=tile_overlap,
    )
    window_size = get_window_size(task)

    # ── 预处理 ──────────────────────────────────────────
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # → NCHW

    # ── 推理 ────────────────────────────────────────────
    try:
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()

            # 反射填充至 window_size 的整数倍
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            output = run_inference(img_lq, model, args, window_size)

        # ── 后处理 ──────────────────────────────────────
        result = tensor_to_image(output, task, h_old, w_old, scale)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'推理失败: {str(e)}')

    # ── 返回图像 ────────────────────────────────────────
    # 将 numpy 图像编码为 PNG 字节流返回
    success, encoded = cv2.imencode('.png', result)
    if not success:
        raise HTTPException(status_code=500, detail='图像编码失败')

    return Response(
        content=encoded.tobytes(),
        media_type='image/png',
        headers={'Content-Disposition': 'inline; filename="enhanced.png"'},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SwinIR Web 推理服务')
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径（.pth 文件）')
    parser.add_argument('--task', type=str, default='real_sr',
                        help='任务类型，默认 real_sr。启动后可通过请求参数覆盖。')
    parser.add_argument('--scale', type=int, default=4,
                        help='超分倍数（默认 4）')
    parser.add_argument('--large_model', action='store_true',
                        help='使用 SwinIR-L 大模型（仅 real_sr 有效）')
    parser.add_argument('--tile', type=int, default=None,
                        help='默认分块大小（不设则整图推理）')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='分块重叠像素数（默认 32）')
    parser.add_argument('--port', type=int, default=8000,
                        help='服务监听端口（默认 8000）')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务监听地址（默认 0.0.0.0，允许外部访问）')
    parser.add_argument('--workers', type=int, default=1,
                        help='worker 进程数（默认 1，多 worker 会各自加载模型）')

    args = parser.parse_args()
    base_args = args

    # 预加载模型
    model, device = load_model(args.model_path)

    print(f'\n{"=" * 60}')
    print(f'  SwinIR Web 推理服务已启动')
    print(f'  地址: http://{args.host}:{args.port}')
    print(f'  API 文档: http://{args.host}:{args.port}/docs')
    print(f'  任务: {args.task}')
    print(f'  设备: {device}')
    print(f'{"=" * 60}\n')

    # 启动 FastAPI（注意：多 worker 模式下每个进程独立加载模型）
    uvicorn.run(
        'web_server:app',
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
    )
