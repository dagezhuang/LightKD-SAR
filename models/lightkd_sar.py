import torch
import torch.nn as nn
from .custom_inverted_residual import CustomInvertedResidual
from .lite_pafpn import LitePAFPN
from .lite_de_head import LiteDecoupledHead
from ..distillation.multi_dim_loss import MultiDimDistillationLoss
from ..distillation.eis_kd import EnhancedInstanceSelection

class LightKDSAR(nn.Module):
    """LightKD-SAR框架"""
    def __init__(self, 
                 num_classes=6,  # SARDet-100k的6个类别：船、飞机、汽车、坦克、桥梁、港口
                 teacher_model=None,  # 教师网络
                 expansion_ratio=4,
                 lambda_kd=0.5):  # 蒸馏损失权重
        super(LightKDSAR, self).__init__()
        self.teacher_model = teacher_model
        self.lambda_kd = lambda_kd
        
        # 1. 轻量级骨干网络（基于自定义反向残差模块）
        self.backbone = nn.Sequential(
            # 初始卷积
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # 反向残差模块组
            CustomInvertedResidual(32,  64, stride=1, expansion_ratio=expansion_ratio),
            CustomInvertedResidual(64,  128, stride=2, expansion_ratio=expansion_ratio),
            CustomInvertedResidual(128, 128, stride=1, expansion_ratio=expansion_ratio),
            CustomInvertedResidual(128, 256, stride=2, expansion_ratio=expansion_ratio),
            CustomInvertedResidual(256, 256, stride=1, expansion_ratio=expansion_ratio),
            CustomInvertedResidual(256, 512, stride=2, expansion_ratio=expansion_ratio),
        )
        
        # 2. 轻量级PAFPN颈部
        self.neck = LitePAFPN(in_channels=[64, 128, 256, 512], out_channels=128)
        
        # 3. 轻量级解耦检测头
        self.head = LiteDecoupledHead(
            num_classes=num_classes,
            in_channels=128,
            feat_channels=64,
            stacked_convs=2
        )
        
        # 4. 知识蒸馏组件
        self.multi_dim_loss = MultiDimDistillationLoss()
        self.eis_module = EnhancedInstanceSelection()
        
        # 冻结教师网络
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def get_backbone_feats(self, x):
        """提取骨干网络的多尺度特征"""
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # 收集特定阶段的特征
            if i in [2, 4, 6, 8]:  # 对应不同尺度的特征输出点
                feats.append(x)
        return feats
    
    def calculate_relations(self, feats):
        """计算特征关系矩阵（用于关系级蒸馏）"""
        relations = []
        for feat in feats:
            b, c, h, w = feat.shape
            feat_flat = feat.view(b, c, -1)  # 展平空间维度
            # 计算特征间余弦相似度作为关系
            relation = F.cosine_similarity(
                feat_flat.unsqueeze(2), 
                feat_flat.unsqueeze(3), 
                dim=1
            ).mean(dim=0)  # 平均批次维度
            relations.append(relation)
        return relations

    def forward_train(self, x, gt_bboxes, gt_labels):
        """训练阶段前向传播"""
        # 学生网络前向
        student_feats = self.get_backbone_feats(x)
        student_fpn_feats = self.neck(student_feats)
        student_cls, student_bbox, student_centerness = self.head(student_fpn_feats)
        
        # 计算学生特征关系
        student_relations = self.calculate_relations(student_fpn_feats)
        
        # 教师网络前向（不更新梯度）
        with torch.no_grad():
            teacher_feats = self.teacher_model.get_backbone_feats(x)
            teacher_fpn_feats = self.teacher_model.neck(teacher_feats)
            teacher_cls, teacher_bbox, _ = self.teacher_model.head(teacher_fpn_feats)
            teacher_relations = self.teacher_model.calculate_relations(teacher_fpn_feats)
        
        # 增强实例选择
        student_preds = {'cls': student_cls[0], 'bbox': student_bbox[0]}
        teacher_preds = {'cls': teacher_cls[0], 'bbox': teacher_bbox[0]}
        selected_mask, selected_weights = self.eis_module(student_preds, teacher_preds, gt_bboxes)
        
        # 计算检测损失（省略具体实现，可根据实际使用的检测框架替换）
        det_loss = self.calculate_detection_loss(
            student_cls, student_bbox, student_centerness,
            gt_bboxes, gt_labels, selected_mask
        )
        
        # 计算蒸馏损失（仅对选中的实例应用）
        kd_loss = self.multi_dim_loss(
            [f[selected_mask] for f in student_fpn_feats],
            [f[selected_mask] for f in teacher_fpn_feats],
            [f[selected_mask] for f in student_cls],
            [f[selected_mask] for f in teacher_cls],
            student_relations,
            teacher_relations
        )
        
        # 总损失 = 检测损失 + 蒸馏损失 * 权重
        total_loss = det_loss + self.lambda_kd * kd_loss
        return total_loss

    def forward_test(self, x):
        """测试阶段前向传播"""
        student_feats = self.get_backbone_feats(x)
        student_fpn_feats = self.neck(student_feats)
        student_cls, student_bbox, student_centerness = self.head(student_fpn_feats)
        
        # 处理预测结果（解码边界框、NMS等）
        results = self.post_process(student_cls, student_bbox, student_centerness)
        return results

    def forward(self, x, gt_bboxes=None, gt_labels=None):
        """统一前向接口"""
        if self.training and gt_bboxes is not None and gt_labels is not None:
            return self.forward_train(x, gt_bboxes, gt_labels)
        else:
            return self.forward_test(x)
    
    def calculate_detection_loss(self, cls_scores, bbox_preds, centernesses, 
                                gt_bboxes, gt_labels, selected_mask):
        """计算检测损失（简化实现）"""
        # 实际使用中应替换为完整的检测损失计算（如Focal Loss + IoU Loss等）
        cls_loss = F.cross_entropy(cls_scores[0][selected_mask], gt_labels[selected_mask])
        bbox_loss = F.l1_loss(bbox_preds[0][selected_mask], gt_bboxes[selected_mask])
        centerness_loss = F.binary_cross_entropy_with_logits(
            centernesses[0][selected_mask].squeeze(), 
            torch.ones_like(gt_labels[selected_mask], dtype=torch.float32)
        )
        return cls_loss + 1.5 * bbox_loss + 0.5 * centerness_loss
    
    def post_process(self, cls_scores, bbox_preds, centernesses):
        """后处理（简化实现）"""
        # 实际使用中应包含边界框解码、置信度筛选、NMS等步骤
        results = []
        for cls_score, bbox_pred, centerness in zip(cls_scores, bbox_preds, centernesses):
            # 简化处理：直接返回前100个高置信度预测
            scores = F.softmax(cls_score, dim=1).max(dim=1)
            topk_indices = scores.values.topk(100).indices
            results.append({
                'scores': scores.values[topk_indices],
                'labels': scores.indices[topk_indices],
                'bboxes': bbox_pred[topk_indices]
            })
        return results
