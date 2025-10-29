#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from ..data_augment import TrainTransform, ValTransform
from ..dataloading import load_image, get_affine_transform

class SARDetDataset(Dataset):
    """SARDet-100K数据集适配"""
    def __init__(self, data_dir, split="train", img_size=(640, 640), preproc=None):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.images_dir = os.path.join(data_dir, split, "images")
        self.labels_dir = os.path.join(data_dir, split, "labels")  # 假设使用YOLO格式txt标注
        
        # 加载所有图像路径
        self.img_files = [f for f in os.listdir(self.images_dir) 
                         if f.endswith((".jpg", ".png", ".jpeg"))]
        self.label_files = [os.path.splitext(f)[0] + ".txt" for f in self.img_files]
        
        # 数据增强
        if preproc is None:
            self.preproc = TrainTransform(
                max_labels=50,
                flip_prob=0.5,
                hsv_prob=0.5
            ) if split == "train" else ValTransform()
        else:
            self.preproc = preproc

        # 类别映射（SARDet-100K的6类）
        self.class_names = ["aircraft", "ship", "car", "tank", "bridge", "harbor"]
        self.class_ids = {name: i for i, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.img_files[index])
        label_path = os.path.join(self.labels_dir, self.label_files[index])
        
        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image {img_path} not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # 加载标注 (class_id, x_center, y_center, w, h)，归一化格式
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path, delimiter=" ").reshape(-1, 5)
            # 转换为绝对坐标 (x1, y1, x2, y2, class_id)
            boxes = labels[:, 1:] * np.array([w, h, w, h])  # 反归一化
            boxes[:, 0] -= boxes[:, 2] / 2  # x_center - w/2 = x1
            boxes[:, 1] -= boxes[:, 3] / 2  # y_center - h/2 = y1
            boxes[:, 2] += boxes[:, 0]      # x1 + w = x2
            boxes[:, 3] += boxes[:, 1]      # y1 + h = y2
            labels = np.hstack([boxes, labels[:, 0:1]]).astype(np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        
        # 数据预处理
        img, labels, _ = self.preproc(img, labels, self.img_size)
        return img, labels, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        """批处理函数"""
        imgs, targets, paths, shapes = zip(*batch)
        targets_list = []
        for i, target in enumerate(targets):
            if len(target) > 0:
                target[:, 4] = i  # 增加batch索引
            targets_list.append(target)
        return (
            torch.stack(imgs),
            torch.cat(targets_list, 0),
            paths,
            shapes
        )