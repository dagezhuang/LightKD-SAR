import torch
import torch.optim as optim
from .eis_kd import EnhancedInstanceSelection
from .multi_dim_loss import MultiDimDistillationLoss

class DistillationTrainer:
    def __init__(self, student_model, teacher_model, 
                 lambda_kd=0.5, lambda_ce=0.5,
                 iou_threshold=0.5, discrepancy_threshold=0.3):
        self.student = student_model
        self.teacher = teacher_model
        self.teacher.eval()  # 教师模型固定
        
        # 损失函数
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.kd_criterion = MultiDimDistillationLoss()
        self.instance_selector = EnhancedInstanceSelection(iou_threshold, discrepancy_threshold)
        
        # 超参数
        self.lambda_kd = lambda_kd  # 蒸馏损失权重
        self.lambda_ce = lambda_ce  # 分类损失权重

    def train_step(self, images, gt_bboxes, gt_labels, optimizer):
        self.student.train()
        with torch.no_grad():
            teacher_preds, teacher_feats, teacher_rels = self.teacher(images)
        
        student_preds, student_feats, student_rels = self.student(images)
        
        # 实例选择
        selected_mask, instance_weights = self.instance_selector(
            student_preds, teacher_preds, gt_bboxes
        )
        
        # 筛选选中的实例
        selected_student_logits = student_preds['cls'][selected_mask]
        selected_teacher_logits = teacher_preds['cls'][selected_mask]
        selected_gt_labels = gt_labels[selected_mask]
        
        # 基础分类损失（仅用选中实例）
        ce_loss = self.cross_entropy(selected_student_logits, selected_gt_labels)
        
        # 多维度蒸馏损失（带实例权重）
        kd_loss = self.kd_criterion(
            student_feats, teacher_feats,
            [selected_student_logits], [selected_teacher_logits],
            student_rels, teacher_rels
        )
        weighted_kd_loss = (kd_loss * instance_weights.mean()).mean()
        
        # 总损失
        total_loss = self.lambda_ce * ce_loss + self.lambda_kd * weighted_kd_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item(),
            'selected_ratio': selected_mask.float().mean().item()
        }