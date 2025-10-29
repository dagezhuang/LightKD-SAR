#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv  # 复用轻量化模块

class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, in_features, depthwise, act)
        self.in_features = in_features
        Conv = DWConv if depthwise else BaseConv

        # 轻量化修改：减少通道数比例（LightKD策略）
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width), int(in_channels[1] * width),
            round(3 * depth), False, depthwise, act
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width), int(in_channels[0] * width),
            round(3 * depth), False, depthwise, act
        )

        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width), int(in_channels[1] * width),
            round(3 * depth), False, depthwise, act
        )

        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width), int(in_channels[2] * width),
            round(3 * depth), False, depthwise, act
        )

        self.out_channels = in_channels

    def forward(self, x):
        # 复用原生特征融合逻辑，内部已使用轻量化CSPLayer
        out_features = self.backbone(x)
        feat1, feat2, feat3 = [out_features[f] for f in self.in_features]

        P5 = self.lateral_conv0(feat3)
        P5_upsample = nn.functional.interpolate(P5, size=feat2.shape[2:], mode="nearest")
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)

        P4 = self.reduce_conv1(P5_upsample)
        P4_upsample = nn.functional.interpolate(P4, size=feat1.shape[2:], mode="nearest")
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out = self.C3_p3(P4_upsample)

        P3_downsample = self.bu_conv2(P3_out)
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        P4_out = self.C3_n3(P3_downsample)

        P4_downsample = self.bu_conv1(P4_out)
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        P5_out = self.C3_n4(P4_downsample)

        return [P3_out, P4_out, P5_out]