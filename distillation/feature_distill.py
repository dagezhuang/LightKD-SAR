import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillation(nn.Module):
    """多尺度特征蒸馏模块，适配YOLOPAFPN的特征层级"""
    def __init__(self, feat_channels=[256, 512, 1024], student_width=0.5, teacher_width=1.0):
        super().__init__()
        # 特征通道对齐（学生模型通道数可能更少）
        self.align_layers = nn.ModuleList()
        for c in feat_channels:
            student_c = int(c * student_width)
            teacher_c = int(c * teacher_width)
            if student_c != teacher_c:
                self.align_layers.append(
                    nn.Conv2d(student_c, teacher_c, kernel_size=1, stride=1, padding=0)
                )
            else:
                self.align_layers.append(nn.Identity())
        
        # 特征相似性损失函数
        self.similarity_loss = nn.MSELoss()  # 或使用余弦相似度

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: 学生模型多尺度特征 [P3, P4, P5]
            teacher_feats: 教师模型多尺度特征 [P3, P4, P5]
        Returns:
            feat_loss: 特征蒸馏总损失
        """
        total_loss = 0.0
        for s_feat, t_feat, align in zip(student_feats, teacher_feats, self.align_layers):
            # 通道对齐 + 尺寸对齐（若特征图大小不同）
            s_feat_align = align(s_feat)
            if s_feat_align.shape[2:] != t_feat.shape[2:]:
                s_feat_align = F.interpolate(
                    s_feat_align, size=t_feat.shape[2:], mode="bilinear", align_corners=False
                )
            # 计算特征损失（可加权不同层级的重要性）
            total_loss += self.similarity_loss(s_feat_align, t_feat)
        
        return total_loss / len(student_feats)  # 平均多尺度损失