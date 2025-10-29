#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from .yolo_pafpn import YOLOPAFPN
from .yolo_head import YOLOXHead

class YOLOX(nn.Module):
    """原生YOLOX模型（支持作为教师模型输出中间特征）"""
    def __init__(self, num_classes=80, depth=1.0, width=1.0, depthwise=False, act="silu"):
        super().__init__()
        # 初始化骨干网络和检测头（避免硬编码，支持动态配置）
        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise, act=act)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise, act=act)

    def forward(self, x, targets=None, need_feats=False):
        """
        新增need_feats参数：控制是否返回中间特征（用于蒸馏）
        """
        fpn_outs = self.backbone(x)  # 先获取骨干网络特征
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "l1_loss": l1_loss,
                "num_fg": num_fg,
            }
            # 若需要特征，附加到输出中
            if need_feats:
                outputs["feats"] = fpn_outs
            return outputs
        else:
            outputs = self.head(fpn_outs)
            # 推理时也可返回特征（如可视化或蒸馏验证）
            if need_feats:
                return outputs, fpn_outs
            return outputs

    def get_backbone_feats(self, x):
        """独立接口：仅获取骨干网络特征（供蒸馏模块调用）"""
        return self.backbone(x)