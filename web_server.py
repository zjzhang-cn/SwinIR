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
import base64
import io
import logging
import os
import queue
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image
from starlette.responses import StreamingResponse

# ── 日志配置 ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('swinir')
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
# 会话管理：{session_id: {'event': threading.Event, 'last_seen': float}}
# 每个浏览器 tab（页面）生成唯一 session_id，同 session 新请求取消上一次推理
sessions = {}
session_lock = threading.Lock()
SESSION_TIMEOUT = 30  # 心跳超时时间（秒），超过此时间无心跳则认为会话已失效


def _get_or_cancel_session(session_id: str):
    """获取 session 的 cancel_event，如果该 session 有正在运行的推理则取消它。"""
    with session_lock:
        now = time.time()
        if session_id in sessions:
            old = sessions[session_id]
            if not old['event'].is_set():
                old['event'].set()
                logger.info(f'取消 session [{session_id[:8]}] 的上一次推理')
        cancel_event = threading.Event()
        sessions[session_id] = {'event': cancel_event, 'last_seen': now}
        logger.info(f'session [{session_id[:8]}] 创建，当前活跃会话: {len(sessions)}')
        return cancel_event


def _update_session_heartbeat(session_id: str):
    """更新 session 的心跳时间戳。"""
    with session_lock:
        if session_id in sessions:
            sessions[session_id]['last_seen'] = time.time()


def _cleanup_stale_sessions():
    """清理超时未收到心跳的会话，取消其正在运行的推理。"""
    with session_lock:
        now = time.time()
        stale = [sid for sid, info in sessions.items()
                 if now - info['last_seen'] > SESSION_TIMEOUT]
        for sid in stale:
            info = sessions.pop(sid)
            if not info['event'].is_set():
                info['event'].set()
                logger.info(f'会话 [{sid[:8]}] 心跳超时 ({SESSION_TIMEOUT}s)，已自动清理')


def _remove_session(session_id: str):
    """session 推理结束后清理。"""
    with session_lock:
        sessions.pop(session_id, None)
        logger.info(f'session [{session_id[:8]}] 已清理，剩余活跃会话: {len(sessions)}')


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
    logger.info(f'计算设备: {dev}')

    # 构建模型配置
    args = create_args(model_path=model_path)
    logger.info(f'正在加载模型: {model_path}')
    m = build_model(args)
    m.eval()
    m = m.to(dev)
    # 统计模型参数量
    total_params = sum(p.numel() for p in m.parameters())
    logger.info(f'模型加载完成 | 参数量: {total_params / 1e6:.1f}M | 设备: {dev}')
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

# 启动事件：加载模型 + 启动会话清理线程
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

    known, _ = parser.parse_known_args()
    base_args = known
    model, device = load_model(known.model_path)

    # 启动后台会话清理线程，每 10 秒检查一次超时会话
    def cleanup_loop():
        while True:
            time.sleep(10)
            _cleanup_stale_sessions()
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info(f'会话清理线程已启动（超时阈值: {SESSION_TIMEOUT}s）')


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


@app.post('/heartbeat')
async def heartbeat(
    session_id: str = Form(..., description='会话 ID'),
):
    """
    心跳接口 — 前端每 5 秒调用一次，告知后端此会话仍然存活。
    如果前端停止心跳（页面关闭/崩溃），后台清理线程会在超时后自动取消该会话的推理。
    """
    _update_session_heartbeat(session_id)
    return {'status': 'ok', 'session': session_id[:8]}


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


def _decode_image(contents: bytes, task: str):
    """将上传的图片字节解码为 numpy 数组 (CHW, float32 [0,1])。"""
    if task in ('gray_dn', 'jpeg_car'):
        pil_img = Image.open(io.BytesIO(contents)).convert('L')
        img = np.array(pil_img).astype(np.float32) / 255.
        img = np.expand_dims(img, axis=0)
    else:
        npy = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npy, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        if img is None:
            raise ValueError('无法解码图像')
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
    return img


def _do_inference(img_lq: np.ndarray, task: str, scale: int, noise: int,
                  jpeg: int, large_model: bool, tile, tile_overlap: int,
                  progress_queue: queue.Queue, cancel_event: threading.Event = None,
                  session_id: str = ''):
    """在后台线程中执行推理，通过 progress_queue 发送进度，通过 cancel_event 支持取消。"""
    sid = session_id[:8] if session_id else '???'
    h, w = img_lq.shape[1], img_lq.shape[2]
    start_time = time.time()
    try:
        logger.info(f'[{sid}] 开始推理 | 任务: {task} | 图像: {w}×{h} | tile: {tile}')
        args = create_args(
            task=task, scale=scale, noise=noise, jpeg=jpeg,
            large_model=large_model, tile=tile, tile_overlap=tile_overlap,
        )
        window_size = get_window_size(task)
        tensor = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, h_old, w_old = tensor.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            tensor = torch.cat([tensor, torch.flip(tensor, [2])], 2)[:, :, :h_old + h_pad, :]
            tensor = torch.cat([tensor, torch.flip(tensor, [3])], 3)[:, :, :, :w_old + w_pad]

            if tile and tile > 0:
                def on_progress(current, total):
                    if cancel_event and cancel_event.is_set():
                        logger.info(f'[{sid}] 推理被取消 (tile {current}/{total})')
                        raise InterruptedError('客户端断开连接')
                    progress_queue.put({'type': 'progress', 'current': current, 'total': total})
                    logger.info(f'[{sid}] 分块推理进度: {current}/{total} ({current*100//total}%)')
                output = run_inference(tensor, model, args, window_size,
                                       progress_callback=on_progress,
                                       cancel_event=cancel_event)
            else:
                output = run_inference(tensor, model, args, window_size)

        if output is None:
            progress_queue.put({'type': 'error', 'message': '推理已取消'})
            return

        result = tensor_to_image(output, task, h_old, w_old, scale)
        success, encoded = cv2.imencode('.png', result)
        if not success:
            raise RuntimeError('图像编码失败')
        img_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        elapsed = time.time() - start_time
        logger.info(f'[{sid}] 推理完成 | 耗时: {elapsed:.2f}s | 输出: {result.shape[1]}×{result.shape[0]}')
        progress_queue.put({'type': 'done', 'image': img_b64})
    except InterruptedError:
        pass  # 已在 on_progress 中记录日志
    except Exception as e:
        logger.error(f'[{sid}] 推理异常: {e}')
        progress_queue.put({'type': 'error', 'message': str(e)})

        result = tensor_to_image(output, task, h_old, w_old, scale)
        success, encoded = cv2.imencode('.png', result)
        if not success:
            raise RuntimeError('图像编码失败')
        img_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        progress_queue.put({'type': 'done', 'image': img_b64})
    except Exception as e:
        progress_queue.put({'type': 'error', 'message': str(e)})


@app.post('/predict-stream')
async def predict_stream(
    image: UploadFile = File(..., description='待增强的输入图像'),
    task: str = Form('real_sr'),
    scale: int = Form(4),
    noise: int = Form(15),
    jpeg:  int = Form(40),
    large_model: bool = Form(False),
    tile: int = Form(None),
    tile_overlap: int = Form(32),
    session_id: str = Form(..., description='会话 ID（每个浏览器 tab 唯一）'),
):
    """分块推理时返回 SSE 进度流，结束时通过 done 事件返回 base64 图像。"""
    if model is None:
        raise HTTPException(status_code=503, detail='模型尚未加载')

    valid_tasks = ['classical_sr', 'lightweight_sr', 'real_sr',
                   'gray_dn', 'color_dn', 'jpeg_car', 'color_jpeg_car']
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f'不支持的任务类型: {task}')

    try:
        contents = await image.read()
        img_lq = _decode_image(contents, task)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'图像读取失败: {str(e)}')

    # 按 session 取消上一次推理，创建新的 cancel_event
    cancel_event = _get_or_cancel_session(session_id)
    logger.info(f'[{session_id[:8]}] 新推理请求 | 任务: {task} | 图像: {img_lq.shape[2]}×{img_lq.shape[1]} | tile: {tile}')

    q = queue.Queue()

    def sse_generator():
        thread = threading.Thread(
            target=_do_inference,
            args=(img_lq, task, scale, noise, jpeg, large_model, tile, tile_overlap, q, cancel_event, session_id),
            daemon=True,
        )
        thread.start()

        try:
            while True:
                try:
                    event = q.get(timeout=1)
                    if event['type'] == 'progress':
                        yield f'data: {{"type":"progress","current":{event["current"]},"total":{event["total"]}}}\n\n'
                    elif event['type'] == 'done':
                        yield f'data: {{"type":"done","image":"data:image/png;base64,{event["image"]}"}}\n\n'
                        yield 'data: {"type":"end"}\n\n'
                        break
                    elif event['type'] == 'error':
                        yield f'data: {{"type":"error","message":"{event["message"]}"}}\n\n'
                        yield 'data: {"type":"end"}\n\n'
                        break
                except queue.Empty:
                    yield ': heartbeat\n\n'
                    if not thread.is_alive():
                        break
        except GeneratorExit:
            cancel_event.set()
            logger.info(f'[{session_id[:8]}] 客户端断开连接，已取消推理')
            raise
        finally:
            _remove_session(session_id)

    return StreamingResponse(
        sse_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


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
    session_id: str = Form(None, description='会话 ID（可选，用于会话管理）'),
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
        img_lq = _decode_image(contents, task)
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

    sid = (session_id or 'no-session')[:8]
    logger.info(f'[{sid}] /predict 请求 | 任务: {task} | 图像: {img_lq.shape[2]}×{img_lq.shape[1]}')
    start_time = time.time()

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
        logger.error(f'[{sid}] 推理失败: {e}')
        raise HTTPException(status_code=500, detail=f'推理失败: {str(e)}')

    # ── 返回图像 ────────────────────────────────────────
    success, encoded = cv2.imencode('.png', result)
    if not success:
        raise HTTPException(status_code=500, detail='图像编码失败')

    elapsed = time.time() - start_time
    logger.info(f'[{sid}] /predict 完成 | 耗时: {elapsed:.2f}s | 输出: {result.shape[1]}×{result.shape[0]}')

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

    logger.info('=' * 50)
    logger.info(f'SwinIR Web 推理服务已启动')
    logger.info(f'地址: http://{args.host}:{args.port}')
    logger.info(f'API 文档: http://{args.host}:{args.port}/docs')
    logger.info(f'Web 页面: http://{args.host}:{args.port}')
    logger.info(f'默认任务: {args.task} | 设备: {device}')
    logger.info('=' * 50)

    # 启动 FastAPI（注意：多 worker 模式下每个进程独立加载模型）
    uvicorn.run(
        'web_server:app',
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
    )
