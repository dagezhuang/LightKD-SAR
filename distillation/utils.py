import torch

def get_backbone_feats(model, x, layers=["dark3", "dark4", "dark5"]):
    """从模型 backbone 提取指定层级的特征（适配YOLOX结构）"""
    feats = model.backbone(x)  # 假设backbone返回字典格式 {layer_name: feat}
    return [feats[layer] for layer in layers]

def log_distill_losses(writer, losses, step):
    """记录蒸馏损失到TensorBoard"""
    writer.add_scalar("distill/total", losses["total"], step)
    writer.add_scalar("distill/feature", losses["feat"], step)
    writer.add_scalar("distill/response", losses["response"], step)
    writer.add_scalar("distill/relation", losses["relation"], step)