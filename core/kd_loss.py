import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    """知识蒸馏损失函数实现"""
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature  # 温度参数控制软化程度
        self.kl_div = nn.KLDivLoss(reduction="batchmean")  # KL散度损失

    def forward(self, student_outputs, teacher_outputs):
        """
        计算学生模型与教师模型的蒸馏损失
        Args:
            student_outputs: 学生模型输出的预测结果 (list of tensor)
            teacher_outputs: 教师模型输出的预测结果 (list of tensor)
        Returns:
            kd_loss: 蒸馏损失值
        """
        total_loss = 0.0
        # 多尺度特征蒸馏（适配YOLOX的3个输出尺度）
        for s_out, t_out in zip(student_outputs, teacher_outputs):
            # 分离类别预测分支 (YOLOX输出格式: [reg, obj, cls])
            s_cls = s_out[:, 5:]  # 类别预测部分
            t_cls = t_out[:, 5:]  # 教师模型类别预测
            
            # 软化概率分布
            s_soft = F.log_softmax(s_cls / self.temperature, dim=1)
            t_soft = F.softmax(t_cls / self.temperature, dim=1)
            
            # 计算KL散度损失（乘以温度平方保持梯度量级）
            total_loss += self.kl_div(s_soft, t_soft) * (self.temperature ** 2)
        
        return total_loss / len(student_outputs)  # 平均多尺度损失