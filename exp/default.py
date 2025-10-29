#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
from .yolox_base import Exp as BaseExp
from distillation.config import DistillConfig  # 导入蒸馏配置
from distillation import (
    FeatureDistillation,
    RelationDistillation,
    MultiDimDistillationLoss
)

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # 基础配置保持与YOLOX-s一致
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 6  # SAR数据集类别数

        # -------------------------- 蒸馏配置扩展 --------------------------
        # 实例化蒸馏超参数管理器
        self.distill_cfg = DistillConfig().update(
            lambda_kd=0.5,          # 总蒸馏损失权重
            lambda_feat=0.3,        # 特征蒸馏权重
            lambda_response=0.2,    # 响应蒸馏权重
            lambda_rel=0.1,         # 关系蒸馏权重
            temperature=4.0,        # 软标签温度
            distill_layers=["dark3", "dark4", "dark5"]  # 蒸馏特征层
        )

        # 教师模型配置
        self.teacher_model = "yolox_l"
        self.teacher_ckpt = "weights/yolox_l.pth"

        # 实例化蒸馏模块（特征蒸馏+关系蒸馏）
        self.feature_distiller = FeatureDistillation(
            feat_channels=[256, 512, 1024],  # 对应YOLOPAFPN输出通道
            student_width=self.width,        # 学生模型宽度系数
            teacher_width=1.0                # 教师模型宽度系数（YOLOX-L为1.0）
        )
        self.relation_distiller = RelationDistillation(normalize=True)

        # 多维度蒸馏损失
        self.distill_loss = MultiDimDistillationLoss(
            distill_cfg=self.distill_cfg,
            feature_distiller=self.feature_distiller,
            relation_distiller=self.relation_distiller
        )

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        # 学生模型：使用轻量化backbone和head
        backbone = YOLOPAFPN(
            depth=self.depth,
            width=self.width,
            depthwise=False,
            act="silu"
        )
        head = YOLOXHead(
            self.num_classes,
            width=self.width,
            depthwise=False,
            act="silu"
        )
        model = YOLOX(backbone, head)
        return model

    def get_trainer(self, args):
        from yolox.core import DistillationTrainer  # 替换为支持多维度蒸馏的Trainer
        from yolox.models import YOLOX as TeacherModel
        from yolox.models import YOLOPAFPN, YOLOXHead

        # 加载教师模型（YOLOX-L配置）
        teacher_backbone = YOLOPAFPN(
            depth=1.0,    # YOLOX-L深度系数
            width=1.0,    # YOLOX-L宽度系数
            depthwise=False,
            act="silu"
        )
        teacher_head = YOLOXHead(
            self.num_classes,
            width=1.0,
            depthwise=False,
            act="silu"
        )
        teacher = TeacherModel(teacher_backbone, teacher_head)
        teacher.load_state_dict(torch.load(self.teacher_ckpt, map_location="cpu"))
        teacher = teacher.to(args.device)
        teacher.eval()  # 教师模型固定为评估模式

        # 返回支持多维度蒸馏的训练器，传入蒸馏损失和配置
        return DistillationTrainer(
            self,
            args,
            teacher,
            distill_loss=self.distill_loss,
            distill_cfg=self.distill_cfg
        )