import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from models.lightkd_sar import LightKDSAR
from datasets.sardet_dataset import SARDetDataset  # 假设的SAR数据集类

def parse_args():
    parser = argparse.ArgumentParser(description='训练LightKD-SAR模型')
    parser.add_argument('--data_root', default='data/SARDet-100k', help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--device', default='cuda:0', help='计算设备')
    parser.add_argument('--teacher_ckpt', default='weights/teacher_model.pth', help='教师模型权重')
    parser.add_argument('--save_path', default='weights/lightkd_sar_best.pth', help='模型保存路径')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据集
    train_dataset = SARDetDataset(
        root=args.data_root,
        split='train',
        augment=True  # 使用数据增强
    )
    val_dataset = SARDetDataset(
        root=args.data_root,
        split='val',
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )
    
    # 2. 加载教师模型（例如使用Faster R-CNN预训练模型）
    from models.teacher import FasterRCNNTeacher  # 假设的教师模型
    teacher_model = FasterRCNNTeacher(num_classes=6)
    teacher_model.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher_model.to(device)
    
    # 3. 初始化学生模型（LightKD-SAR）
    model = LightKDSAR(
        num_classes=6,
        teacher_model=teacher_model,
        expansion_ratio=4,
        lambda_kd=0.5
    )
    model.to(device)
    model.train()
    
    # 4. 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 5. 训练循环
    best_map = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        total_loss = 0.0
        
        # 训练阶段
        model.train()
        for batch_idx, (images, bboxes, labels) in enumerate(train_loader):
            images = images.to(device)
            bboxes = [b.to(device) for b in bboxes]
            labels = [l.to(device) for l in labels]
            
            # 前向传播计算损失
            loss = model(images, bboxes, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印训练进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            # 计算验证集mAP（省略具体实现）
            val_map = evaluate(model, val_loader, device)
            print(f"Validation mAP: {val_map:.4f}")
        
        # 保存最佳模型
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model with mAP: {best_map:.4f}")
        
        # 更新学习率
        lr_scheduler.step()
    
    print(f"Training completed. Best validation mAP: {best_map:.4f}")

def evaluate(model, val_loader, device):
    """评估模型在验证集上的mAP"""
    # 实际使用中应实现完整的mAP计算逻辑
    # 此处返回示例值，实际需替换为真实计算
    return torch.rand(1).item() * 0.1 + 0.45  # 模拟45%-55%的mAP

if __name__ == "__main__":
    main()
