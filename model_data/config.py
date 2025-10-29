import os

# 训练集路径
TRAIN_PATH = os.path.join(os.path.dirname(__file__), '../dataset/SARDet-100K/')
TRAIN_ANNOT_PATH = os.path.join(os.path.dirname(__file__), '../dataset/SARDet-100K/')

# 验证集路径
VAL_PATH = os.path.join(os.path.dirname(__file__), '../dataset/SARDet-100K/val')
VAL_ANNOT_PATH = os.path.join(os.path.dirname(__file__), '../dataset/SARDet-100K/')

# 类别数（与classes.txt一致，保持不变）
NUM_CLASSES = 6

# 输入图像尺寸
INPUT_SHAPE = (640, 640)

# 锚框路径（保持不变）
ANCHORS_PATH = os.path.join(os.path.dirname(__file__), 'sardet100k_anchors.txt')

# 预训练权重路径（学生模型初始化用，保持不变）
PRETRAINED_WEIGHTS = os.path.join(os.path.dirname(__file__), 'yolox_s_sardet.pth')

# -------------------------- 新增：蒸馏相关路径配置 --------------------------
# 教师模型权重路径（用于知识蒸馏）
TEACHER_WEIGHTS = os.path.join(os.path.dirname(__file__), 'yolox_l_sardet.pth')  # 假设教师模型为YOLOX-L

# 蒸馏配置文件路径（可选，若需从文件加载蒸馏超参数）
DISTILL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'distill_config')