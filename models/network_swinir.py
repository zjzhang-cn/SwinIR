# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------
#
# ==============================================================================
# SwinIR 架构总览（三阶段流水线）：
#
#   输入图像 (B, C, H, W)
#       │
#       ▼
#   ┌──────────────────────────────────────────────┐
#   │  第一阶段：浅层特征提取 (shallow feature)       │
#   │  conv_first: Conv2d(in_chans → embed_dim)    │
#   │  提取低级特征（边缘、纹理等）                    │
#   └──────────────────────────────────────────────┘
#       │
#       ▼
#   ┌──────────────────────────────────────────────┐
#   │  第二阶段：深层特征提取 (deep feature)          │
#   │  ┌─ RSTB × N ─────────────────────────────┐  │
#   │  │  BasicLayer (多个 SwinTransformerBlock)  │  │
#   │  │       ↓                                  │  │
#   │  │  Conv2d (残差连接)                        │  │
#   │  └──────────────────────────────────────────┘  │
#   │  conv_after_body: 最后的卷积层                  │
#   └──────────────────────────────────────────────┘
#       │
#       ▼
#   ┌──────────────────────────────────────────────┐
#   │  第三阶段：高质量图像重建 (reconstruction)       │
#   │  根据 upsampler 类型选择不同的重建路径：          │
#   │  - pixelshuffle: 经典 SR（多级 PixelShuffle）  │
#   │  - pixelshuffledirect: 轻量 SR（单级）         │
#   │  - nearest+conv: 真实世界 SR（最近邻+卷积）     │
#   │  - ""/None: 去噪/JPEG去伪影（无上采样）        │
#   └──────────────────────────────────────────────┘
#       │
#       ▼
#   输出图像 (B, C, H*scale, W*scale)
#
# 关键设计思想：
# 1. 窗口注意力 (W-MSA)：在局部窗口内计算自注意力，将复杂度从 O(H²W²)
#    降为 O(window_size² × HW)，使 Transformer 能处理高分辨率图像
# 2. 移位窗口 (SW-MSA)：相邻层之间窗口偏移 window_size/2，实现跨窗口
#    信息交换，弥补窗口注意力的感受野局限
# 3. 相对位置编码：学习窗口内 token 之间的相对位置偏置，比绝对位置编码
#    更适合视觉任务
# 4. RSTB 残差连接：每个 Swin Transformer 组通过卷积残差连接，融合
#    Transformer 的全局建模能力和卷积的局部归纳偏置
#
# 组件说明：
# - Mlp: 两层全连接前馈网络（FFN），带 GELU 激活和 Dropout
# - window_partition / window_reverse: 窗口划分与还原的工具函数
# - WindowAttention: 基于窗口的多头自注意力（W-MSA），含相对位置偏置
# - SwinTransformerBlock: Swin Transformer 基本块，交替使用 W-MSA/SW-MSA
# - PatchMerging: 补丁合并层（SwinIR 中不使用下采样，该组件保留但未启用）
# - BasicLayer: 一个阶段的多个 SwinTransformerBlock 堆叠，奇偶交替 W-MSA/SW-MSA
# - RSTB: 残差 Swin Transformer 块，BasicLayer + Conv 残差连接
# - PatchEmbed / PatchUnEmbed: 序列(空间)格式与图像格式之间的相互转换
# - Upsample / UpsampleOneStep: PixelShuffle 上采样模块
# - SwinIR: 顶层模型类，组装三阶段流水线
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .swinir_components import PatchEmbed, PatchUnEmbed, RSTB, Upsample, UpsampleOneStep, trunc_normal_
except ImportError:
    from swinir_components import PatchEmbed, PatchUnEmbed, RSTB, Upsample, UpsampleOneStep, trunc_normal_

class SwinIR(nn.Module):
    r"""SwinIR 主模型 —— 基于 Swin Transformer 的图像修复

    三阶段流水线：
      1. 浅层特征提取：conv_first（3×3 卷积）
      2. 深层特征提取：多个 RSTB（残差 Swin Transformer 块）+ conv_after_body
      3. 高质量图像重建：根据 upsampler 类型使用不用的重建路径

    支持的任务和对应配置：
      - classical_sr：经典超分辨率 → pixelshuffle
      - lightweight_sr：轻量超分辨率 → pixelshuffledirect
      - real_sr：真实世界超分辨率 → nearest+conv
      - gray_dn/color_dn：灰度/彩色去噪 → 无上采样（upsampler=""）
      - jpeg_car/color_jpeg_car：JPEG 去伪影 → 无上采样（upsampler=""）

    参数:
        img_size: 输入图像尺寸（默认 64）
        patch_size: 补丁大小（默认 1，即逐像素处理）
        in_chans: 输入图像通道数（3=RGB, 1=灰度）
        embed_dim: 特征嵌入维度（Transformer 的隐藏维度）
        depths: 每个 RSTB 中 SwinTransformerBlock 的数量
        num_heads: 每个 RSTB 的注意力头数
        window_size: 窗口注意力大小（7 或 8）
        mlp_ratio: MLP 隐藏层扩展比率
        qkv_bias: QKV 线性层是否加偏置
        qk_scale: QK 缩放因子
        drop_rate: Dropout 比率
        attn_drop_rate: 注意力 Dropout 比率
        drop_path_rate: 随机深度比率
        norm_layer: 归一化层类型
        ape: 是否使用绝对位置编码（SwinIR 中默认 False）
        patch_norm: 是否在 patch embedding 后进行归一化
        use_checkpoint: 是否使用梯度检查点
        upscale: 放大倍数（SR 任务 >1，去噪/去伪影 =1）
        img_range: 图像像素值范围（1.0 或 255.0）
        upsampler: 重建模块类型 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: RSTB 残差连接卷积类型 '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        # RGB 图像的 ImageNet 均值，用于输入归一化
        # 去噪/灰度任务均值为 0（不做均值减法）
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        # 第一阶段：浅层特征提取
        # 单个 3×3 卷积，将输入图像映射到 embed_dim 维特征空间
        # 提取边缘、纹理等低级特征，为后续 Transformer 深层处理做准备
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        # 第二阶段：深层特征提取
        # 核心部分：多个 RSTB 堆叠，每个 RSTB 内部包含：
        #   BasicLayer (SwinTransformerBlock × depth) + Conv 残差连接
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # 将特征图切分为不重叠的补丁（patch_size=1 时即逐像素处理）
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 将补丁序列合并还原为图像（PatchEmbed 的逆操作）
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # 绝对位置编码（SwinIR 中默认不使用，相对位置编码已足够）
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减策略：从 0 线性增长到 drop_path_rate
        # 深层 block 有更高的丢弃概率，起到正则化作用
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建 RSTB 块序列
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # 深层特征提取的最后卷积层（RSTB 输出后的精炼层）
        if resi_connection == '1conv':
            # 单个 3×3 卷积
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 瓶颈结构：dim → dim/4 → dim/4 → dim
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        # 第三阶段：高质量图像重建
        # 根据 upsampler 配置选择不同的重建路径
        if self.upsampler == 'pixelshuffle':
            # 经典 SR 路径（classical_sr）：
            # conv_before_upsample → 多级 PixelShuffle 上采样 → conv_last
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # 轻量 SR 路径（lightweight_sr）：
            # 直接单步 PixelShuffle，省去中间的 conv_before_upsample
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # 真实世界 SR 路径（real_sr）：
            # conv_before_upsample → 最近邻插值上采样 → 卷积细化 → conv_last
            # 使用最近邻插值而非 PixelShuffle，减少块状伪影
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                # 4 倍放大需要两次 2 倍最近邻上采样
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # 去噪/JPEG去伪影路径（upsampler="" 或 None）：
            # 无上采样，直接 conv_last 输出
            # 采用残差学习：output = input + conv_last(deep_features)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """排除权重衰减的参数：绝对位置编码不使用 L2 正则化"""
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """排除权重衰减的参数关键词：相对位置偏置表不使用 L2 正则化"""
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        """检查并填充输入图像尺寸为 window_size 的整数倍

        Swin Transformer 的窗口划分要求 H 和 W 都能被 window_size 整除。
        使用反射填充 (reflection padding) 补足不足部分，
        推理完成后在 forward 中裁剪回原始尺寸。
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        """深层特征提取的前向传播

        流程：PatchEmbed → [AbsolutePosEmbed] → Dropout → RSTB × N → Norm → PatchUnEmbed
        返回空间格式的特征图 (B, C, H, W)。
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # 空间 → 序列：(B, C, H, W) → (B, H*W, C)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C —— 最终的 LayerNorm
        x = self.patch_unembed(x, x_size)  # 序列 → 空间：(B, H*W, C) → (B, C, H, W)

        return x

    def forward(self, x):
        """SwinIR 完整前向传播

        统一处理流程（所有任务共有）：
        1. check_image_size：反射填充到 window_size 整数倍
        2. 输入归一化：(x - mean) * img_range  将像素值映射到模型工作范围
        3. 浅层特征提取 → 深层特征提取 → 图像重建
        4. 反归一化：x / img_range + mean
        5. 裁剪回原始尺寸

        不同 upsampler 对应不同重建子路径（见对应代码注释）。
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # 输入归一化：减去均值后乘以范围因子
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # 经典 SR：conv_first → 深层特征(残差连接) → conv_before_upsample → Upsample → conv_last
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # 轻量 SR：conv_first → 深层特征(残差连接) → UpsampleOneStep（直接输出 RGB）
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # 真实世界 SR：conv_first → 深层特征(残差连接) → 最近邻插值 → 卷积细化 → conv_last
            # 分步上采样减少块状伪影，适合真实世界退化图像
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # 去噪/JPEG 去伪影：残差学习方式
            # output = input + conv_last(deep_features)
            # 模型学习的是噪声/伪影残差，而非直接重建干净图像
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        # 反归一化：将像素值还原到原始范围
        x = x / self.img_range + self.mean

        # 裁剪回原始尺寸（去除 check_image_size 添加的反射填充）
        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    # 测试代码：创建轻量 SR 模型并进行一次前向推理
    # 参数对应 lightweight_sr 配置：embed_dim=60, depths=[6,6,6,6], mlp_ratio=2, upsampler='pixelshuffledirect'
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)

