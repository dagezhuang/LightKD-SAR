"""知识蒸馏超参数配置"""
class KDConfig:
    def __init__(self):
        # 蒸馏温度（控制概率分布软化程度）
        self.temperature = 2.0
        # 蒸馏损失权重（平衡原始损失与蒸馏损失）
        self.lambda_kd = 0.5
        # 特征蒸馏层级（YOLOX的3个输出尺度）
        self.distill_layers = [0, 1, 2]
        # 教师模型冻结模式（是否冻结BN层）
        self.freeze_teacher_bn = True
        # 蒸馏开始的epoch（可选：先训练学生模型再蒸馏）
        self.start_distill_epoch = 0

    def parse_from_dict(self, config_dict):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self