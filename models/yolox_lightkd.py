#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch.nn as nn
from .yolox import YOLOX
from .yolo_pafpn import YOLOPAFPN
from .yolo_head import YOLOXHead

class YOLOXLightKD(YOLOX):
    """轻量化学生模型（基于LightKD策略）"""
    def __init__(self, num_classes=6, depth=0.33, width=0.50, depthwise=False, act="silu"):
        # 强制使用轻量化配置：更小的深度/宽度系数
        super().__init__(
            num_classes=num_classes,
            depth=depth,
            width=width,
            depthwise=depthwise,
            act=act
        )
        # 可额外添加学生模型特有的轻量化修改（如通道剪枝、层缩减）
        self._init_light_kd()

    def _init_light_kd(self):
        """LightKD特有的轻量化初始化（如替换部分卷积为深度可分离卷积）"""
        # 示例：将检测头的部分卷积替换为DWConv（根据需求可选）
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                # 仅替换3x3卷积为深度可分离卷积（保持轻量化）
                m = nn.Sequential(
                    nn.Conv2d(m.in_channels, m.in_channels, 3, 1, 1, groups=m.in_channels, bias=False),
                    nn.Conv2d(m.in_channels, m.out_channels, 1, 1, 0, bias=True)
                )

    def forward(self, x, targets=None, need_feats=False):
        # 复用父类YOLOX的forward逻辑，确保与教师模型接口一致
        return super().forward(x, targets, need_feats=need_feats)