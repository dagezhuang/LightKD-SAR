import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力机制用于抑制SAR斑点噪声"""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
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
        return x * y.expand_as(x)  # 对通道特征进行加权，抑制噪声通道

class CustomInvertedResidual(nn.Module):
    """SAR适配的自定义反向残差模块（图1右半部分实现）"""
    def __init__(self, in_channels, out_channels, stride, expansion_ratio=4):
        super(CustomInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "步长必须为1或2"
        
        # 低代价扩展操作 L(C(·))
        self.low_cost_expansion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_ratio, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * expansion_ratio),
            nn.ReLU6(inplace=True)
        )
        
        # 深度可分离卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels * expansion_ratio, 
                      in_channels * expansion_ratio,
                      kernel_size=3, stride=stride, padding=1, 
                      groups=in_channels * expansion_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expansion_ratio),
            nn.ReLU6(inplace=True)
        )
        
        # SE注意力机制（抑制斑点噪声）
        self.se_layer = SELayer(in_channels * expansion_ratio)
        
        # 投影操作
        self.project_conv = nn.Sequential(
            nn.Conv2d(in_channels * expansion_ratio, out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 特征重用分支（跳跃连接）
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        # 共享输入下采样（当步长为2时）
        if stride == 2:
            self.downsample = nn.Conv2d(in_channels, out_channels, 
                                       kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        # 主分支
        out = self.low_cost_expansion(x)
        out = self.depthwise_conv(out)
        out = self.se_layer(out)  # 应用SE注意力抑制噪声
        out = self.project_conv(out)
        
        # 特征重用与下采样处理
        if self.use_res_connect:
            residual = x
        elif self.downsample is not None:
            residual = self.downsample(x)  # 共享输入下采样
        else:
            residual = 0
            
        return out + residual  # 特征融合
