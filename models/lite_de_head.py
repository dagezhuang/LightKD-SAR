import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_inverted_residual import CustomInvertedResidual

class LiteDecoupledHead(nn.Module):
    """轻量级解耦检测头用于SAR多类别目标检测"""
    def __init__(self, num_classes, in_channels=128, feat_channels=64, stacked_convs=2):
        super(LiteDecoupledHead, self).__init__()
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        
        # 分类分支（轻量级卷积堆叠）
        self.cls_convs = nn.ModuleList()
        # 回归分支（轻量级卷积堆叠）
        self.reg_convs = nn.ModuleList()
        
        for i in range(stacked_convs):
            # 分类分支使用自定义反向残差模块
            self.cls_convs.append(
                CustomInvertedResidual(
                    in_channels if i == 0 else feat_channels,
                    feat_channels,
                    stride=1,
                    expansion_ratio=2
                )
            )
            # 回归分支使用自定义反向残差模块
            self.reg_convs.append(
                CustomInvertedResidual(
                    in_channels if i == 0 else feat_channels,
                    feat_channels,
                    stride=1,
                    expansion_ratio=2
                )
            )
        
        # 分类输出（多类别预测）
        self.cls_score = nn.Conv2d(feat_channels, num_classes, kernel_size=3, padding=1)
        # 回归输出（中心坐标+宽高）
        self.bbox_pred = nn.Conv2d(feat_channels, 4, kernel_size=3, padding=1)
        # 中心-ness输出（目标中心置信度）
        self.centerness = nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1)

    def forward(self, feats):
        """
        Args:
            feats: 多尺度特征列表 [P3, P4, P5, P6]
        Returns:
            cls_scores: 分类预测列表
            bbox_preds: 回归预测列表
            centernesses: 中心-ness预测列表
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        
        for x in feats:
            # 分类分支
            cls_feat = x
            for conv in self.cls_convs:
                cls_feat = conv(cls_feat)
            cls_score = self.cls_score(cls_feat)
            
            # 回归分支
            reg_feat = x
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat)
            bbox_pred = self.bbox_pred(reg_feat)
            centerness = self.centerness(reg_feat)
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
        
        return cls_scores, bbox_preds, centernesses
