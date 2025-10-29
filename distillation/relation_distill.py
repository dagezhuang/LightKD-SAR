import torch
import torch.nn as nn

class RelationDistillation(nn.Module):
    """实例间关系蒸馏，对齐师生模型的目标关联模式"""
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize  # 是否对特征归一化

    def forward(self, student_feats, teacher_feats):
        """
        Args:
            student_feats: 学生模型的实例特征 (N, C)
            teacher_feats: 教师模型的实例特征 (N, C)
        Returns:
            rel_loss: 关系蒸馏损失
        """
        if self.normalize:
            # 特征归一化（增强稳定性）
            student_feats = F.normalize(student_feats, dim=1)
            teacher_feats = F.normalize(teacher_feats, dim=1)
        
        # 计算实例间相似度矩阵（N×N）
        s_sim = torch.mm(student_feats, student_feats.t())  # 学生特征相似度
        t_sim = torch.mm(teacher_feats, teacher_feats.t())  # 教师特征相似度
        
        # 对齐相似度分布
        return nn.MSELoss()(s_sim, t_sim)