import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedInstanceSelection(nn.Module):
    """增强实例选择机制，聚焦SAR稀疏目标"""
    def __init__(self, iou_threshold=0.5, discrepancy_threshold=0.3, 
                 min_area_ratio=0.001, background_threshold=0.1,background_class=-1):
        super(EnhancedInstanceSelection, self).__init__()
        self.iou_threshold = iou_threshold  # IoU阈值
        self.discrepancy_threshold = discrepancy_threshold  # 差异阈值
        self.min_area_ratio = min_area_ratio  # 最小目标面积占比（相对图像）
        self.background_threshold = background_threshold  # 背景置信度阈值
        self.background_class = background_class  # -1表示无背景类

    def calculate_discrepancy(self, student_pred, teacher_pred):
        """计算师生预测差异 D_I^r，新增置信度加权"""
        # 分类分数差异（增加置信度加权）
        t_conf = torch.max(F.softmax(teacher_pred['cls'], dim=1), dim=1)[0]  # 教师置信度
        cls_diff = F.l1_loss(student_pred['cls'], teacher_pred['cls'], reduction='none').mean(dim=1)
        weighted_cls_diff = cls_diff * t_conf  # 高置信度教师预测的差异更重要
        
        # 边界框IoU差异（1 - IoU）
        bbox_iou = self.bbox_iou(student_pred['bbox'], teacher_pred['bbox'])
        bbox_diff = 1 - bbox_iou
        
        # 综合差异（动态调整权重）
        discrepancy = 0.7 * weighted_cls_diff + 0.3 * bbox_diff  # 提高分类差异权重
        return discrepancy

    def bbox_iou(self, bboxes1, bboxes2):
        """计算边界框IoU，修复空框处理"""
        # bboxes格式: [x1, y1, x2, y2]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        # 处理面积为0的边界框（避免除以0）
        union = area1 + area2 - inter
        union = torch.clamp(union, min=1e-6)
        iou = inter / union
        return iou

    def is_valid_target(self, bboxes, img_h, img_w):
        """筛选有效目标（排除过小目标和背景）"""
        # 计算目标面积占比
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        img_area = img_h * img_w
        area_ratio = bbox_areas / img_area
        
        # 排除过小目标
        valid_area = area_ratio > self.min_area_ratio
        
        # 排除背景框（假设最后一类是背景）
        # 这里假设cls的最后一个维度是背景概率
        background_prob = F.softmax(bboxes.get('cls', torch.zeros_like(bboxes[:, :1])), dim=1)[:, -1]
        valid_foreground = background_prob < self.background_threshold

        if self.background_class != -1:
            # 仅当指定了背景类时才筛选
            background_prob = F.softmax(bboxes['cls'], dim=1)[:, self.background_class]
            valid_foreground = background_prob < self.background_threshold
        else:
            valid_foreground = torch.ones_like(valid_area, dtype=torch.bool)  # 不筛选背景
        return valid_area & valid_foreground

    def forward(self, student_preds, teacher_preds, gt_bboxes, img_shape):
        """
        Args:
            student_preds: 学生网络预测 (cls, bbox)
            teacher_preds: 教师网络预测 (cls, bbox)
            gt_bboxes: 真实边界框
            img_shape: 图像尺寸 (h, w) - 新增参数，用于筛选小目标
        Returns:
            selected_mask: 选中实例的掩码
            selected_weights: 选中实例的权重
        """
        # 1. 筛选有效目标（排除过小目标和背景）
        img_h, img_w = img_shape
        valid_mask = self.is_valid_target(teacher_preds, img_h, img_w)
        
        # 2. 计算师生预测差异（仅对有效目标计算）
        discrepancy = self.calculate_discrepancy(student_preds, teacher_preds)
        discrepancy = discrepancy * valid_mask.float()  # 无效目标差异置为0
        
        # 3. 计算教师预测与真实框的IoU
        teacher_gt_iou = self.bbox_iou(teacher_preds['bbox'], gt_bboxes)
        
        # 4. 第一阶段筛选：高差异实例（结合有效性）
        high_discrepancy_mask = (discrepancy > self.discrepancy_threshold) & valid_mask
        
        # 5. 第二阶段筛选：教师高置信实例
        high_teacher_iou_mask = teacher_gt_iou > self.iou_threshold
        
        # 6. 最终选中的实例（逻辑与）
        selected_mask = high_discrepancy_mask & high_teacher_iou_mask
        
        # 7. 改进实例权重计算（增加IoU加权）
        # 教师预测越准（IoU越高）且差异越大的实例权重越高
        selected_weights = torch.where(
            selected_mask,
            (discrepancy / discrepancy.max()) * teacher_gt_iou,  # 双重加权
            torch.zeros_like(discrepancy)
        )
        
        # 防止权重为0导致的梯度问题
        selected_weights = selected_weights + 1e-8
        
        return selected_mask, selected_weights

class EISKDLoss:
    """Enhanced Intersection Similarity Knowledge Distillation Loss"""
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def bbox_iou(self, bboxes1, bboxes2):
        """计算边界框IoU，修复空框处理"""
        # bboxes格式: [x1, y1, x2, y2]，形状为[N,4]和[M,4]
        if bboxes1.numel() == 0 or bboxes2.numel() == 0:
            return torch.tensor(0.0, device=bboxes1.device)

        # 计算面积
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])  # [N]
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])  # [M]

        # 计算交集坐标
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [N,M,2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]，确保不出现负值
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        # 计算并集和IoU（避免除以0）
        union = area1[:, None] + area2 - inter  # [N,M]
        union = torch.clamp(union, min=1e-6)
        iou = inter / union  # [N,M]
        return iou

    def forward(self, student_preds, teacher_preds):
        """
        计算学生与教师预测框的蒸馏损失
        student_preds: 学生模型输出，形状为[B, N, 4+1+num_classes]
        teacher_preds: 教师模型输出，形状为[B, M, 4+1+num_classes]
        """
        total_loss = 0.0
        batch_size = student_preds.shape[0]

        for b in range(batch_size):
            # 提取当前批次的预测框（过滤低置信度）
            student_boxes = student_preds[b][student_preds[b, :, 4] > 0.2, :4]  # [N,4]
            teacher_boxes = teacher_preds[b][teacher_preds[b, :, 4] > 0.5, :4]   # [M,4]

            if student_boxes.numel() == 0 or teacher_boxes.numel() == 0:
                continue  # 无有效框时跳过

            # 计算IoU矩阵
            iou_matrix = self.bbox_iou(student_boxes, teacher_boxes)  # [N,M]
            max_iou, _ = iou_matrix.max(dim=1)  # 每个学生框与最佳教师框的IoU

            # 计算蒸馏损失（让学生框尽可能接近教师框）
            loss = 1 - max_iou.mean()
            total_loss += loss

        return total_loss / batch_size if batch_size > 0 else 0.0
