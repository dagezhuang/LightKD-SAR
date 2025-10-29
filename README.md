# LightKD-SAR: Lightweight Architecture with Knowledge Distillation for High-Performance SAR Object Detection

## 项目简介
LightKD-SAR 是一个基于 YOLOX 改进的轻量级 SAR（合成孔径雷达）目标检测框架，通过知识蒸馏（Knowledge Distillation）技术在保证检测精度的同时显著降低模型计算成本，适用于资源受限的 SAR 图像处理场景。

## 核心特性
- **轻量化设计**：通过通道剪枝、简化检测头等策略减少模型参数量和计算量
- **多维度知识蒸馏**：融合特征蒸馏、响应蒸馏和关系蒸馏，从教师模型迁移知识
- **适配 SAR 数据**：针对 SAR 图像特性优化数据处理流程，支持 SARDet-100K 数据集

## 安装依赖
`pip install -r requirements.txt`

### 模型训练
`bash`
`python train.py -f exp/default.py -d 1 -b 16 --fp16`
`python -m torch.distributed.launch --nproc_per_node=2 train.py -f exp/default.py -d 2 -b 32 --fp16`

### 模型评估
`python get_map.py`

### 参数分析
`python summary.py`


如有问题，请联系  zhouzhuang@csu.ac.cn


