class DistillConfig:
    """知识蒸馏超参数配置"""
    def __init__(self):
        # 损失权重
        self.lambda_kd = 0.5        # 蒸馏总权重
        self.lambda_feat = 0.3      # 特征蒸馏权重
        self.lambda_response = 0.2  # 响应蒸馏权重
        self.lambda_rel = 0.1       # 关系蒸馏权重（若使用）
        
        # 温度参数（用于软化概率分布）
        self.temperature = 4.0
        
        # 实例选择参数（增强EIS模块的可配置性）
        self.iou_threshold = 0.5
        self.discrepancy_threshold = 0.3
        self.min_area_ratio = 0.001  # SAR目标筛选阈值
        
        # 特征蒸馏层级
        self.distill_layers = ["dark3", "dark4", "dark5"]  # 对应YOLOPAFPN的输出
        
    def update(self, **kwargs):
        """动态更新配置参数"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self