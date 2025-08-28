import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDimDistillationLoss(nn.Module):
    """多维度知识蒸馏损失（特征级+关系级+响应级）"""
    def __init__(self, lambda_feature=0.5, lambda_relation=0.3, lambda_response=0.2):
        super(MultiDimDistillationLoss, self).__init__()
        self.lambda_feature = lambda_feature  # 特征级损失的权重
        self.lambda_relation = lambda_relation  # 关系级损失权重
        self.lambda_response = lambda_response  # 响应级损失权重
        
        # L1损失用于特征蒸馏
        self.feature_criterion = nn.L1Loss()
        # 交叉熵损失用于响应蒸馏
        self.response_criterion = nn.CrossEntropyLoss()

    def forward(self, student_feats, teacher_feats, 
                student_logits, teacher_logits, 
                student_relations, teacher_relations):
        """
        Args:
            student_feats: 学生网络特征列表
            teacher_feats: 教师网络特征列表
            student_logits: 学生网络分类输出
            teacher_logits: 教师网络分类输出
            student_relations: 学生网络特征关系矩阵
            teacher_relations: 教师网络特征关系矩阵
        Returns:
            total_loss: 多维度蒸馏总损失
        """
        # 1. 特征级蒸馏损失（对齐师生特征分布）
        feature_loss = 0.0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            # 调整学生特征尺寸以匹配教师特征
            s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], 
                                  mode='bilinear', align_corners=True)
            feature_loss += self.feature_criterion(s_feat, t_feat)
        
        # 2. 关系级蒸馏损失（对齐特征间关系）
        relation_loss = 0.0
        for s_rel, t_rel in zip(student_relations, teacher_relations):
            # 计算关系矩阵的MSE损失
            relation_loss += F.mse_loss(s_rel, t_rel)
        
        # 3. 响应级蒸馏损失（对齐分类输出）
        response_loss = 0.0
        for s_logit, t_logit in zip(student_logits, teacher_logits):
            # 教师输出软化处理
            t_logit_soft = F.softmax(t_logit / 4.0, dim=1)  # 温度系数T=4
            response_loss += F.kl_div(F.log_softmax(s_logit / 4.0, dim=1), 
                                     t_logit_soft, reduction='batchmean') * (4.0 ** 2)
        
        # 总蒸馏损失
        total_loss = (self.lambda_feature * feature_loss +
                     self.lambda_relation * relation_loss +
                     self.lambda_response * response_loss)
        
        return total_loss
