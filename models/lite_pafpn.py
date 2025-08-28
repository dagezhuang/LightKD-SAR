import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_inverted_residual import CustomInvertedResidual

class LitePAFPN(nn.Module):
    """轻量级路径聚合特征金字塔网络处理SAR多尺度目标检测"""
    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=128):
        super(LitePAFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 上采样路径（自顶向下）
        self.lateral_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        # 下采样路径（自底向上）
        self.down_convs = nn.ModuleList()
        
        # 构建横向连接和上采样层
        for i in range(len(in_channels)-1, -1, -1):
            # 横向卷积（调整通道数）
            self.lateral_convs.append(
                nn.Conv2d(in_channels[i], out_channels, kernel_size=1, stride=1, padding=0)
            )
            if i > 0:
                # 上采样后使用轻量级反向残差模块融合特征
                self.up_convs.append(
                    CustomInvertedResidual(out_channels, out_channels, stride=1, expansion_ratio=2)
                )
        
        # 构建下采样路径
        for i in range(len(in_channels)-1):
            self.down_convs.append(
                CustomInvertedResidual(out_channels, out_channels, stride=2, expansion_ratio=2)
            )
        
        # 输出卷积（进一步融合特征）
        self.out_convs = nn.ModuleList([
            CustomInvertedResidual(out_channels, out_channels, stride=1)
            for _ in range(len(in_channels))
        ])

    def forward(self, inputs):
        """
        Args:
            inputs: 骨干网络输出的多尺度特征 [C3, C4, C5, C6]
        Returns:
            融合后的多尺度特征 [P3, P4, P5, P6]
        """
        assert len(inputs) == len(self.in_channels)
        
        # 自顶向下路径
        laterals = [
            lateral_conv(inputs[i]) 
            for i, lateral_conv in enumerate(reversed(self.lateral_convs))
        ]
        
        # 上采样融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1):
            # 上采样到相同尺寸
            laterals[i + 1] += F.interpolate(
                laterals[i], scale_factor=2.0, mode='bilinear', align_corners=True
            )
            # 使用轻量级模块优化融合特征
            laterals[i + 1] = self.up_convs[i](laterals[i + 1])
        
        # 自底向上路径
        for i in range(used_backbone_levels - 1):
            laterals[i] += self.down_convs[i](laterals[i + 1])
        
        # 输出特征优化
        outputs = [
            self.out_convs[i](laterals[i]) 
            for i in range(used_backbone_levels)
        ]
        
        return outputs[::-1]  # 反转以匹配输入顺序
