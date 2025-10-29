import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import get_lr

class LightKDTrainer:
    """LightKD蒸馏训练器"""
    def __init__(self, model, teacher_model, kd_loss, lambda_kd=0.5):
        self.model = model  # 学生模型
        self.teacher_model = teacher_model  # 教师模型
        self.kd_loss = kd_loss  # 蒸馏损失函数
        self.lambda_kd = lambda_kd  # 蒸馏损失权重
        self.criterion = nn.CrossEntropyLoss()  # 原始任务损失

    def train_one_epoch(self, epoch, train_loader, optimizer, device, scaler=None):
        self.model.train()
        self.teacher_model.eval()  # 教师模型固定
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_kd_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # 学生模型输出
                student_outputs = self.model(images)
                # 教师模型输出（不计算梯度）
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                
                # 计算原始任务损失（以分类损失为例，需根据实际任务调整）
                cls_loss = self._calc_task_loss(student_outputs, targets)
                # 计算蒸馏损失
                distill_loss = self.kd_loss(student_outputs, teacher_outputs)
                # 总损失 = 原始损失 + 蒸馏损失*权重
                loss = cls_loss + self.lambda_kd * distill_loss
            
            # 反向传播
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_kd_loss += distill_loss.item()
            
            pbar.set_postfix({
                "lr": get_lr(optimizer),
                "total_loss": total_loss / (pbar.n + 1),
                "cls_loss": total_cls_loss / (pbar.n + 1),
                "kd_loss": total_kd_loss / (pbar.n + 1)
            })
        
        return {
            "total_loss": total_loss / len(train_loader),
            "cls_loss": total_cls_loss / len(train_loader),
            "kd_loss": total_kd_loss / len(train_loader)
        }
    
    def _calc_task_loss(self, outputs, targets):
        """计算原始检测任务损失（需与YOLOX损失函数适配）"""
        # 此处需根据YOLOX的损失计算逻辑实现
        # 示例：假设outputs包含分类预测，targets为类别标签
        cls_preds = torch.cat([out[:, 5:] for out in outputs], dim=0)  # 聚合所有尺度的类别预测
        cls_targets = targets[:, 0].long()  # 假设targets第一列为类别ID
        return self.criterion(cls_preds, cls_targets)