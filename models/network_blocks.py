#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        return SiLU()
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError(f"Unsupported act type: {name}")

# LightKD轻量化注意力
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# BaseConv
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=ksize, 
            stride=stride, padding=pad, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# DWConv
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize, stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, 1, 1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))

# LightKD的CustomInvertedResidual
class CustomInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=2.0, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(in_channels * expansion)  # 倒残差升维
        Conv = DWConv if depthwise else BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)  # 1x1升维
        self.conv2 = Conv(hidden_channels, hidden_channels, 3, stride=1, act=act)  # 3x3特征提取
        self.se = SELayer(hidden_channels)  # 嵌入SE注意力
        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)  # 1x1降维

        self.use_add = shortcut and in_channels == out_channels  # 保持原生残差逻辑

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)  # 应用注意力
        y = self.conv3(y)
        return y + x if self.use_add else y

# 使用CustomInvertedResidual
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, 1, act=act)
        
        # 核心修改：用CustomInvertedResidual替换Bottleneck
        self.m = nn.Sequential(*[
            CustomInvertedResidual(
                hidden_channels, hidden_channels, shortcut, 2.0, depthwise, act=act
            ) for _ in range(n)
        ])

    def forward(self, x):
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))

# 原生SPPBottleneck保持不变
class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(ks, 1, ks//2) for ks in kernel_sizes])
        self.conv2 = BaseConv(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1, 1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))