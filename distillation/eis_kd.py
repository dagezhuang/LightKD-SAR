import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedInstanceSelection(nn.Module):
    """增强实例选择机制，聚焦SAR稀疏目标"""
    def __init__(self, iou_threshold=0.5, discrepancy_threshold=0.3):
        super(EnhancedInstanceSelection, self).__init__()
        self.iou_threshold = iou_threshold  # IoU阈值
        self.discrepancy_threshold = discrepancy_threshold  # 差异阈值

    def calculate_discrepancy(self, student_pred, teacher_pred):
        """计算师生预测差异 D_I^r"""
        # 分类分数差异
        cls_diff = F.l1_loss(student_pred['cls'], teacher_pred['cls'], reduction='none').mean(dim=1)
        
        # 边界框IoU差异（1 - IoU）
        bbox_iou = self.bbox_iou(student_pred['bbox'], teacher_pred['bbox'])
        bbox_diff = 1 - bbox_iou
        
        # 综合差异（加权求和）
        discrepancy = 0.6 * cls_diff + 0.4 * bbox_diff
        return discrepancy

    def bbox_iou(self, bboxes1, bboxes2):
        """计算边界框IoU"""
        # bboxes格式: [x1, y1, x2, y2]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        iou = inter / (area1 + area2 - inter + 1e-6)
        return iou

    def forward(self, student_preds, teacher_preds, gt_bboxes):
        """
        Args:
            student_preds: 学生网络预测 (cls, bbox)
            teacher_preds: 教师网络预测 (cls, bbox)
            gt_bboxes: 真实边界框
        Returns:
            selected_mask: 选中实例的掩码
            selected_weights: 选中实例的权重
        """
        # 计算师生预测差异
        discrepancy = self.calculate_discrepancy(student_preds, teacher_preds)
        
        # 计算教师预测与真实框的IoU
        teacher_gt_iou = self.bbox_iou(teacher_preds['bbox'], gt_bboxes)
        
        # 第一阶段筛选：高差异实例
        high_discrepancy_mask = discrepancy > self.discrepancy_threshold
        
        # 第二阶段筛选：教师高置信实例
        high_teacher_iou_mask = teacher_gt_iou > self.iou_threshold
        
        # 最终选中的实例（逻辑与）
        selected_mask = high_discrepancy_mask & high_teacher_iou_mask
        
        # 计算实例权重（差异越大权重越高）
        selected_weights = torch.where(
            selected_mask,
            discrepancy / discrepancy.max(),  # 归一化差异作为权重
            torch.zeros_like(discrepancy)
        )
        
        return selected_mask, selected_weights
