import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """多层感知机（MLP / FFN - 前馈网络）

    Swin Transformer 中每个注意力层之后的前馈网络。
    结构：Linear → GELU → Dropout → Linear → Dropout
    隐藏层维度通常是输入维度的 mlp_ratio 倍（默认 4 倍）。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """窗口划分：将特征图切分为不重叠的窗口

    这是 Swin Transformer 的核心操作之一。将 (B, H, W, C) 的特征图
    重塑为 (B×num_windows, window_size, window_size, C) 的窗口序列，
    使后续的自注意力计算限制在每个小窗口内，大幅降低计算复杂度。

    参数:
        x: 输入张量，形状为 (B, H, W, C)
        window_size: 窗口大小（整数，窗口为正方形）

    返回:
        windows: 窗口序列，形状为 (num_windows*B, window_size, window_size, C)

    计算步骤：
    1. view 将 H/W 维度按 window_size 切分：B × (H/ws) × ws × (W/ws) × ws × C
    2. permute 调换维度顺序，使每个窗口的 ws×ws 区域连续
    3. view 合并 batch 和窗口数量维度
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """窗口还原：将窗口序列还原为原始特征图

    window_partition 的逆操作。在窗口内完成注意力计算后，
    将窗口序列重新拼回完整的 (B, H, W, C) 特征图。

    参数:
        windows: 窗口序列，形状为 (num_windows*B, window_size, window_size, C)
        window_size: 窗口大小
        H: 原始特征图高度
        W: 原始特征图宽度

    返回:
        x: 还原后的特征图，形状为 (B, H, W, C)

    计算步骤：
    1. 从窗口总数反推 batch_size
    2. view 将窗口序列按网格排列
    3. permute 恢复正确的空间维度顺序
    4. view 还原为 (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""基于窗口的多头自注意力（W-MSA / Window Multi-head Self-Attention）

    这是 Swin Transformer 的核心注意力模块。与标准 Transformer 的全局注意力不同，
    W-MSA 仅在每个 window_size × window_size 的局部窗口内计算注意力，将计算复杂度从
    O((HW)²C) 降为 O(window_size² × HW × C)，使 Transformer 能够处理高分辨率图像。

    关键特性：
    1. 相对位置偏置 (Relative Position Bias)：为窗口内每对 token 学习一个偏置项，
       增强模型对空间位置关系的感知能力
    2. 支持移位窗口 (Shifted Window)：通过 attn_mask 实现循环移位后的窗口注意力，
       使跨窗口信息交换成为可能

    参数:
        dim: 输入通道数
        window_size: 窗口的高和宽（元组）
        num_heads: 注意力头数
        qkv_bias: 是否给 QKV 线性层添加偏置
        qk_scale: 覆盖默认的 QK 缩放因子（head_dim ** -0.5）
        attn_drop: 注意力权重的 Dropout 比率
        proj_drop: 输出投影的 Dropout 比率
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表：为窗口内所有可能的相对位置对存储一个可学习的偏置值
        # 相对位置范围为 [-(Wh-1), Wh-1] × [-(Ww-1), Ww-1]，共 (2Wh-1)*(2Ww-1) 种
        # 形状：(2*Wh-1)*(2*Ww-1), nH —— 每种相对位置在每个注意力头有独立的偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 构建窗口内每对 token 的相对位置索引
        # 例如 window_size=8 时，共有 64 个 token，产生 64×64=4096 对关系
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 广播相减得到所有 token 对的相对坐标：(2, Wh*Ww, 1) - (2, 1, Wh*Ww) → (2, Wh*Ww, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 将相对坐标从 [-Wh+1, Wh-1] 平移到 [0, 2Wh-2]，方便索引
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 将二维相对坐标压缩为一维索引：(h_idx * (2Ww-1) + w_idx)
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 注册为 buffer（不参与梯度更新，保存到 state_dict）
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV 联合投影：一次线性变换同时生成 Query、Key、Value
        # 输入 dim 维，输出 3*dim 维（分别对应 Q、K、V）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 用截断正态分布初始化相对位置偏置表
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征，形状为 (num_windows*B, N, C)
               其中 N = window_size * window_size，即每个窗口内的 token 数
            mask: 注意力掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
                  用于移位窗口注意力（SW-MSA），屏蔽不该互相注意的 token 对
        """
        B_, N, C = x.shape
        # QKV 投影 + 重塑：(B_, N, 3*dim) → (3, B_, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别取出 Query、Key、Value

        # 缩放 Query，防止点积过大导致 softmax 梯度消失
        q = q * self.scale
        # 计算注意力分数：Q @ K^T → (B_, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1))

        # 从偏置表中查表获取相对位置偏置，并加到注意力分数上
        # relative_position_index.view(-1) 将 (Wh*Ww, Wh*Ww) 展平为一维索引
        # 查表后 reshape 为 (Wh*Ww, Wh*Ww, nH)，再 permute 为 (nH, Wh*Ww, Wh*Ww)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # SW-MSA 模式：应用注意力掩码
            # 掩码将不属于同一区域的 token 对的注意力分数设为 -100（softmax 后 ≈ 0）
            nW = mask.shape[0]  # 窗口数量
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            # W-MSA 模式：直接 softmax
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # 注意力加权求和：(B_, num_heads, N, N) @ (B_, num_heads, N, head_dim) → (B_, num_heads, N, head_dim)
        # transpose + reshape → (B_, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # 计算 1 个窗口（包含 N 个 token）的浮点运算量
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer 基本块

    每个块包含：
    1. LayerNorm → W-MSA 或 SW-MSA（取决于 shift_size）
    2. 残差连接 + DropPath
    3. LayerNorm → MLP（FFN）
    4. 残差连接 + DropPath

    移位窗口机制（核心创新）：
    - 偶数块 (shift_size=0)：标准窗口注意力（W-MSA），窗口从特征图左上角开始
    - 奇数块 (shift_size=window_size//2)：移位窗口注意力（SW-MSA），窗口偏移半个窗口大小
    - 实现方式：通过 torch.roll 循环移位特征图，而非重新划分窗口，避免窗口数量增加

    Cyclic Shift + Mask 原理：
    1. 将特征图循环移位 shift_size 个像素
    2. 按标准窗口划分
    3. 在注意力计算时使用 mask 屏蔽原本不相邻的区域
    4. 计算完成后反向循环移位还原

    参数:
        dim: 输入通道数
        input_resolution: 输入特征图的分辨率 (H, W)
        num_heads: 注意力头数
        window_size: 窗口大小
        shift_size: SW-MSA 的偏移量（0 为 W-MSA，>0 为 SW-MSA）
        mlp_ratio: MLP 隐藏层维度与嵌入维度的比率
        qkv_bias: 是否给 QKV 线性层添加偏置
        qk_scale: QK 缩放因子
        drop: Dropout 比率
        attn_drop: 注意力 Dropout 比率
        drop_path: 随机深度 (Stochastic Depth) 比率
        act_layer: 激活函数
        norm_layer: 归一化层
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # 如果输入分辨率小于窗口大小，则不进行窗口划分，直接做全局注意力
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # DropPath：随机深度正则化，训练时以 drop_path 概率随机丢弃整个残差分支
        # drop_path=0 时使用 nn.Identity() 避免额外计算开销
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # SW-MSA 模式下预计算注意力掩码，W-MSA 模式下无需掩码
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """计算 SW-MSA 的注意力掩码

        循环移位后，特征图的 4 个角和边缘被移到一起。如果直接做窗口注意力，
        原本不相邻的像素会被错误地互相注意到。掩码的作用是将这些跨区域 token
        对的注意力分数设为 -100（softmax 后 ≈ 0），从而屏蔽它们。

        实现方式：
        1. 创建 img_mask，将移位后的特征图按窗口划分成 3×3=9 个区域，标记 0-8
        2. 窗口划分后，同一窗口内不同区域的 token 对的掩码值 ≠0
        3. 将这些位置的注意力分数设为 -100（近似 -inf）
        """
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        # 将特征图按窗口边界分为 3×3=9 个区域（见 Swin Transformer 论文图 4）
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # 广播相减：同一区域 token 差值为 0，不同区域 token 差值 ≠0
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 将不同区域的 token 对设为 -100（→ softmax ≈ 0），同区域保持 0
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # 序列格式 → 空间格式：准备做循环移位

        # 循环移位：将特征图沿 H 和 W 方向各移动 shift_size 个像素
        # 移出的像素从对面边界"循环"回来，保持张量尺寸不变
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 划分为窗口并做注意力
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # 当输入分辨率与构造函数中的 resolution 匹配时，使用预计算的掩码
        # 不匹配时（分块推理），动态计算掩码
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 合并窗口还原为特征图
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 反向循环移位：将特征图恢复到原始位置
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)  # 空间格式 → 序列格式

        # 双残差连接：W-MSA/SW-MSA 残差 + MLP 残差
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""补丁合并层（Patch Merging）

    将 2×2 网格内的 4 个相邻补丁合并为一个补丁，实现 2 倍下采样。
    每个 2×2 区域内的 4 个 token 沿通道维度拼接（C → 4C），
    然后通过线性层降维到 2C。

    注意：SwinIR 不做下采样（这是超分辨率任务，需要保持或增加分辨率），
    因此该组件在 SwinIR 中保留但实际不使用（downsample=None）。
    它来自原始 Swin Transformer 的分类架构。

    参数:
        input_resolution: 输入特征图的分辨率
        dim: 输入通道数
        norm_layer: 归一化层
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # 按步长 2 采样 2×2 网格的 4 个位置
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C  —— 左上
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C  —— 左下
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C  —— 右上
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C  —— 右下
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C —— 沿通道拼接
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # 4C → 2C 降维

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """Swin Transformer 的基本层（一个阶段）

    由多个 SwinTransformerBlock 按顺序堆叠而成，
    偶数块使用 W-MSA（shift_size=0），奇数块使用 SW-MSA（shift_size=window_size//2），
    实现窗口内和跨窗口信息的交替传播。

    支持梯度检查点 (activation checkpointing) 以节省 GPU 显存：
    在前向传播中不保存中间激活值，反向传播时重新计算。

    参数:
        dim: 输入通道数
        input_resolution: 输入分辨率
        depth: SwinTransformerBlock 的数量
        num_heads: 注意力头数
        window_size: 局部窗口大小
        mlp_ratio: MLP 隐藏维度比率
        qkv_bias: QKV 偏置
        qk_scale: QK 缩放因子
        drop: Dropout 比率
        attn_drop: 注意力 Dropout 比率
        drop_path: 随机深度比率（可为 float 或 list，list 时逐块递增）
        norm_layer: 归一化层
        downsample: 层末尾的下采样模块（SwinIR 中为 None）
        use_checkpoint: 是否使用梯度检查点节省显存
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建 SwinTransformerBlock 序列
        # 偶数块 shift_size=0（W-MSA），奇数块 shift_size=window_size//2（SW-MSA）
        # 这种交替设计确保每一层都能在不同窗口之间交换信息
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # 补丁合并下采样（SwinIR 中不使用，downsample 始终为 None）
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                # 梯度检查点：不保存中间激活值，反向时重新计算前向
                # 以计算时间换取 GPU 显存
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """残差 Swin Transformer 块（Residual Swin Transformer Block）

    RSTB 是 SwinIR 深层特征提取的核心构建块。每个 RSTB 包含：
    1. BasicLayer：多个 SwinTransformerBlock（奇偶交替 W-MSA/SW-MSA）
    2. Conv2d 卷积层：一个 3×3 卷积（或 3 个连续的瓶颈卷积）
    3. 残差连接：Conv(BasicLayer(x)) + x

    设计思想：
    Transformer 的全局建模能力 + 卷积的局部归纳偏置，
    通过残差连接融合两者优势。

    卷积残差连接的两种模式：
    - '1conv'：单个 3×3 卷积，参数少，效率高（默认）
    - '3conv'：瓶颈结构（1×1 降维 → 3×3 卷积 → 1×1 升维），参数量更少

    参数:
        dim: 输入通道数
        input_resolution: 输入分辨率
        depth: BasicLayer 中的 SwinTransformerBlock 数量
        num_heads: 注意力头数
        window_size: 窗口大小
        mlp_ratio: MLP 扩展比率
        qkv_bias: QKV 偏置
        qk_scale: QK 缩放
        drop: Dropout 比率
        attn_drop: 注意力 Dropout 比率
        drop_path: 随机深度比率
        norm_layer: 归一化层
        downsample: 下采样模块
        use_checkpoint: 是否使用梯度检查点
        img_size: 输入图像尺寸
        patch_size: 补丁大小
        resi_connection: 残差连接类型 '1conv' 或 '3conv'
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        # 卷积残差连接：增强局部特征提取能力
        if resi_connection == '1conv':
            # 单个 3×3 卷积，参数：dim × dim × 3 × 3
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 瓶颈结构：dim → dim/4 → dim/4 → dim，减少参数量
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # PatchEmbed/PatchUnEmbed 用于序列 ↔ 空间格式转换
        # in_chans=0 表示不需要输入通道的投影（仅做格式转换）
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        # 核心公式：output = Conv(BasicLayer(x)) + x
        # 1. residual_group: Transformer 处理
        # 2. patch_unembed: 序列 → 空间格式 (B, H*W, C) → (B, C, H, W)
        # 3. conv: 3×3 卷积增强局部特征
        # 4. patch_embed: 空间 → 序列格式 (B, C, H, W) → (B, H*W, C)
        # 5. + x: 残差连接
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r"""图像到补丁序列的嵌入（Patch Embedding）

    将空间特征图 (B, C, H, W) 转换为序列格式 (B, H*W, C)，使后续
    Transformer 层可以处理。这一步只做格式转换（flatten + transpose），
    不做额外的卷积投影（原始 Swin Transformer 中会有，但 SwinIR 中
    卷积投影由 conv_first 完成）。

    参数:
        img_size: 输入图像尺寸（默认 224）
        patch_size: 补丁大小（默认 4，但 SwinIR 中通常为 1）
        in_chans: 输入通道数（SwinIR 中设为 0，表示直接格式转换无投影）
        embed_dim: 嵌入维度
        norm_layer: 可选的归一化层
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 将空间特征图展平为序列：(B, C, H, W) → (B, C, H*W) → (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r"""补丁序列到图像的还原（Patch Unembedding）

    PatchEmbed 的逆操作。将序列格式 (B, H*W, C) 还原为空间特征图 (B, C, H, W)。
    在 RSTB 中用于将 Transformer 输出转换回空间格式，以进行卷积残差连接。

    参数:
        img_size: 图像尺寸
        patch_size: 补丁大小
        in_chans: 输入通道数
        embed_dim: 嵌入维度
        norm_layer: 可选的归一化层
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        # 序列格式还原为空间特征图：(B, H*W, C) → (B, C, H, W)
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """多级 PixelShuffle 上采样模块

    用于经典超分辨率 (classical SR) 的 upsample 路径。
    对于 2^n 倍放大，堆叠 n 对 (Conv2d → PixelShuffle(2))，每对放大 2 倍。
    对于 3 倍放大，使用单次 (Conv2d → PixelShuffle(3))。

    PixelShuffle 原理：将通道维度的像素重新排列到空间维度。
    例如 Conv2d(C → 4C) + PixelShuffle(2) 将 (4C, H, W) → (C, 2H, 2W)。

    参数:
        scale: 放大倍数，支持 2^n 和 3
        num_feat: 中间特征通道数
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale 是 2 的幂次（2, 4, 8...）
            for _ in range(int(math.log(scale, 2))):
                # 每次将通道数扩大 4 倍，然后 PixelShuffle 将空间分辨率扩大 2 倍
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            # 3 倍放大：通道数扩大 9 倍，PixelShuffle(3) 将空间分辨率扩大 3 倍
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """单步 PixelShuffle 上采样模块

    用于轻量超分辨率 (lightweight SR) 的 upsample 路径。
    与 Upsample 的区别：无论放大多少倍，始终只用一次 Conv2d + 一次 PixelShuffle，
    减少了参数和计算量，适合移动端/轻量部署。

    例如 scale=4 时，Upsample 需要 2×Conv+2×PixelShuffle，
    而 UpsampleOneStep 只需 1×Conv(→16C)+1×PixelShuffle(4)。

    参数:
        scale: 放大倍数，支持 2^n 和 3
        num_feat: 输入特征通道数
        num_out_ch: 输出通道数（通常等于输入图像通道数）
        input_resolution: 输入分辨率（用于 FLOPs 计算）
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops



