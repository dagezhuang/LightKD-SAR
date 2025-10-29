#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from .network_blocks import (
    BaseConv, DWConv, Focus, SPPBottleneck,
    CSPLayer, CustomInvertedResidual  # 引入新模块
)

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        # 基础通道和深度（保持原生配置）
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # 原生stem保持不变
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 各阶段网络（保持结构，内部已通过CSPLayer使用新残差块）
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act)
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act)
        )

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act)
        )

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x  # 80x80
        x = self.dark4(x)
        outputs["dark4"] = x  # 40x40
        x = self.dark5(x)
        outputs["dark5"] = x  # 20x20
        return {k: v for k, v in outputs.items() if k in self.out_features}