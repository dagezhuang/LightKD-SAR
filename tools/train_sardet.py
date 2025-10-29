#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import torch
from torch.utils.data import DataLoader
from yolox.data.datasets import SARDetDataset
from yolox.models.yolox_lightkd import LightKDYOLOX
from yolox.models.yolox import YOLOX as TeacherModel
from yolox.core import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/SARDet-100K")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--teacher_ckpt", default="weights/yolox_l.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载数据集
    train_dataset = SARDetDataset(
        data_dir=args.data_dir,
        split="train",
        img_size=(640, 640)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=SARDetDataset.collate_fn
    )
    
    # 2. 初始化教师模型
    teacher_model = TeacherModel()
    teacher_model.load_state_dict(torch.load(args.teacher_ckpt, map_location=args.device))
    teacher_model.to(args.device)
    teacher_model.eval()
    
    # 3. 初始化学生模型（带蒸馏）
    student_model = LightKDYOLOX(
        num_classes=6,
        teacher_model=teacher_model,
        lambda_kd=0.5
    )
    student_model.to(args.device)
    student_model.train()
    
    # 4. 启动训练
    trainer = Trainer(
        model=student_model,
        train_loader=train_loader,
        device=args.device,
        epochs=args.epochs
    )
    trainer.train()

if __name__ == "__main__":
    main()