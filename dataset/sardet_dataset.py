import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

class SARDetDataset(Dataset):
    def __init__(self, root, split='train', augment=False):
        self.root = root
        self.split = split
        self.augment = augment
        self.images_dir = os.path.join(root, 'images', split)
        self.annotations_dir = os.path.join(root, 'annotations', split)
        self.image_ids = [f.split('.')[0] for f in os.listdir(self.images_dir)]
        
        # 类别映射（SARDet-100k包含6类目标）
        self.classes = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f'{img_id}.png')
        ann_path = os.path.join(self.annotations_dir, f'{img_id}.xml')
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 加载标注
        bboxes, labels = self._parse_annotation(ann_path)
        
        # 数据增强（简化实现）
        if self.augment:
            image, bboxes = self._augment(image, bboxes)
        
        # 转换为Tensor
        image = self._to_tensor(image)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, bboxes, labels

    def _parse_annotation(self, ann_path):
        """解析XML标注文件获取边界框和类别"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in self.class_to_idx:
                continue  # 跳过未定义类别
            
            # 获取边界框 (xmin, ymin, xmax, ymax)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])
            
            # 生成类别标签（one-hot编码）
            label = [0.] * len(self.classes)
            label[self.class_to_idx[cls]] = 1.
            labels.append(label)
        
        return bboxes, labels

    def _to_tensor(self, image):
        """将PIL图像转换为Tensor并归一化"""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def _augment(self, image, bboxes):
        """简单数据增强（随机翻转和旋转）"""
        # 实际实现需添加数据增强逻辑
        return image, bboxes

    def collate_fn(self, batch):
        """自定义批处理函数，处理不同数量的目标"""
        images = torch.stack([item[0] for item in batch])
        bboxes = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        return images, bboxes, labels