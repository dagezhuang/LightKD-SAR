#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from .network_blocks import BaseConv, DWConv  # 复用轻量化模块

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False, lightweight=True):
        super().__init__()
        self.num_classes = num_classes
        self.width = width
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        # 轻量化修改：减少检测头通道数（LightKD策略）
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            # 特征降维（比原生减少25%通道）
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width),
                         out_channels=int(256 * width // 1.25),  # 轻量化
                         ksize=1, stride=1, act=act)
            )
            # 减少卷积层数（LightKD轻量化）
            self.cls_convs.append(nn.Sequential(
                Conv(in_channels=int(256 * width // 1.25),
                     out_channels=int(256 * width // 1.25),
                     ksize=3, stride=1, act=act),
            ))
            self.reg_convs.append(nn.Sequential(
                Conv(in_channels=int(256 * width // 1.25),
                     out_channels=int(256 * width // 1.25),
                     ksize=3, stride=1, act=act),
            ))
            self.cls_preds.append(
                nn.Conv2d(int(256 * width // 1.25), num_classes, 1, 1, 0)
            )
            self.reg_preds.append(
                nn.Conv2d(int(256 * width // 1.25), 4, 1, 1, 0)
            )
            self.obj_preds.append(
                nn.Conv2d(int(256 * width // 1.25), 1, 1, 1, 0)
            )

    def forward(self, inputs):
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            # 分类分支
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)
            # 回归分支
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            # 输出拼接
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs