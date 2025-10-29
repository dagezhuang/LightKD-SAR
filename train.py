#-------------------------------------#
#       对sardet100k数据集进行训练（带蒸馏）
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

# 蒸馏相关组件
from core.trainer import LightKDTrainer
from distillation.config import DistillConfig
from distillation.multi_dim_loss import MultiDimDistillationLoss
from distillation.feature_distill import FeatureDistillation
from distillation.relation_distill import RelationDistillation
from models.yolox import YOLOX as TeacherModel
from models.yolo_pafpn import YOLOPAFPN
from models.yolo_head import YOLOXHead

if __name__ == "__main__":
    #-------------------------------#
    # 基础配置（保持原逻辑）
    #-------------------------------#
    Cuda            = True
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = False
    classes_path    = 'model_data/sardet100k_classes.txt'
    model_path      = 'model_data/yolox_s.pth'  # 学生模型初始权重
    teacher_path    = 'model_data/yolox_l.pth'  # 教师模型权重
    input_shape     = [640, 640]
    phi             = 's'
    mosaic          = True
    mosaic_prob     = 0.5
    mixup           = True
    mixup_prob      = 0.5
    special_aug_ratio = 0.7

    #-------------------------------#
    # 蒸馏配置（明确参数）
    #-------------------------------#
    use_distillation    = True
    distill_cfg = DistillConfig().update(
        lambda_kd=0.5,          # 蒸馏总权重
        lambda_feat=0.3,        # 特征蒸馏权重
        lambda_response=0.2,    # 响应蒸馏权重
        lambda_rel=0.1,         # 关系蒸馏权重
        temperature=4.0,        # 软标签温度
        distill_layers=["dark3", "dark4", "dark5"]
    )

    #-------------------------------#
    # 训练参数（明确优化器和学习率策略）
    #-------------------------------#
    # 冻结/解冻训练参数
    Init_Epoch          = 0
    Freeze_Epoch        = 30
    Freeze_batch_size   = 8
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Freeze_Train        = True

    # 优化器配置（明确参数）
    optimizer_type      = "sgd"  # 可选 "adam" 或 "sgd"
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    weight_decay        = 5e-4  # 权重衰减系数
    momentum            = 0.937  # SGD动量参数

    # 学习率策略（明确类型和参数）
    lr_decay_type       = "cos"  # 可选 "cos" 或 "step"
    lr_steps            = [50, 80]  # 若为step策略，设置衰减节点
    lr_gamma            = 0.1      # 若为step策略，衰减系数

    # 其他配置
    train_annotation_path   = 'SARDet-100K/train.txt'
    val_annotation_path     = 'SARDet-100K/val.txt'
    log_dir                 = 'logs/sardet/'
    save_period             = 10
    eval_flag               = True
    eval_period             = 10
    warmup_period           = 5
    VOCdevkit_path          = 'SARDet-100K'

    # 设置随机种子
    seed_everything(seed)

    # 获取类别
    class_names, num_classes = get_classes(classes_path)
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')

    #-------------------------------#
    # 初始化学生模型
    #-------------------------------#
    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        print(f'加载学生模型权重: {model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    #-------------------------------#
    # 初始化教师模型（固定参数）
    #-------------------------------#
    if use_distillation:
        print(f'加载教师模型权重: {teacher_path}')
        teacher_backbone = YOLOPAFPN(depth=1.0, width=1.0)  # YOLOX-L配置
        teacher_head = YOLOXHead(num_classes, width=1.0)
        teacher_model = TeacherModel(teacher_backbone, teacher_head)
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    else:
        teacher_model = None

    #-------------------------------#
    # 多卡配置
    #-------------------------------#
    if distributed:
        local_rank  = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device      = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        model_train = model.to(device)
        model_train = torch.nn.parallel.DistributedDataParallel(
            model_train, device_ids=[local_rank], find_unused_parameters=True
        )
    else:
        model_train = model.to(device)

    #-------------------------------#
    # 损失函数（原始任务损失+蒸馏损失）
    #-------------------------------#
    # 原始检测损失（YOLO损失）
    yolo_loss = YOLOLoss(num_classes, fp16)

    # 蒸馏损失（多维度：特征+响应+关系）
    if use_distillation:
        feature_distiller = FeatureDistillation(
            feat_channels=[256, 512, 1024],
            student_width=0.5,  # YOLOX-s宽度系数
            teacher_width=1.0   # YOLOX-L宽度系数
        )
        relation_distiller = RelationDistillation(normalize=True)
        kd_loss = MultiDimDistillationLoss(
            distill_cfg=distill_cfg,
            feature_distiller=feature_distiller,
            relation_distiller=relation_distiller
        )
    else:
        kd_loss = None

    #-------------------------------#
    # 数据加载
    #-------------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #-------------------------------#
    # 优化器（明确参数）
    #-------------------------------#
    pg0, pg1, pg2 = [], [], []  # 分组参数（bn/bias/weight）
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # 偏置参数（不衰减）
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # BN参数（不衰减）
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # 权重参数（衰减）

    # 选择优化器并配置参数
    if optimizer_type == "adam":
        optimizer = optim.Adam(
            pg0, lr=Init_lr, betas=(0.9, 0.999)
        )
    else:  # sgd
        optimizer = optim.SGD(
            pg0, lr=Init_lr, momentum=momentum, nesterov=True
        )
    # 添加其他参数组
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    #-------------------------------#
    # 学习率调度器（明确策略）
    #-------------------------------#
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type=lr_decay_type,
        lr=Init_lr,
        min_lr=Min_lr,
        total_iters=UnFreeze_Epoch,
        warmup_iters=warmup_period,
        lr_steps=lr_steps,  # step策略专用
        lr_gamma=lr_gamma   # step策略专用
    )

    #-------------------------------#
    # 训练工具（日志、评估、EMA）
    #-------------------------------#
    # 日志记录
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(log_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # 评估回调
    eval_callback = EvalCallback(
        model, input_shape, num_classes, val_lines, val_annotation_path,
        VOCdevkit_path, log_dir, Cuda, eval_flag=eval_flag, period=eval_period
    )

    # 模型EMA（指数移动平均）
    ema = ModelEMA(model_train) if not distributed else None

    # 断点续训
    if Init_Epoch != 0:
        model_path = os.path.join(log_dir, f"ep{Init_Epoch:03d}-loss0-val_loss0.pth")
        if os.path.exists(model_path):
            print(f'加载断点权重: {model_path}')
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            if ema:
                ema.ema.load_state_dict(torch.load(model_path, map_location=device))
                ema.updates = Init_Epoch * num_train // batch_size

    #-------------------------------#
    # 冻结训练配置
    #-------------------------------#
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    # 自适应调整学习率（根据batch_size）
    nbs = 64  # 标称batch_size
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 0.1), lr_limit_max * 0.1)
    set_optimizer_lr(optimizer, Init_lr_fit)

    #-------------------------------#
    # 数据加载器
    #-------------------------------#
    train_dataset = YoloDataset(
        train_lines, input_shape, num_classes, mosaic=mosaic,
        mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
        special_aug_ratio=special_aug_ratio, train=True
    )
    val_dataset = YoloDataset(
        val_lines, input_shape, num_classes, mosaic=False,
        mixup=False, train=False
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // dist.get_world_size()
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(
        train_dataset, shuffle=shuffle, batch_size=batch_size,
        num_workers=4, pin_memory=True, sampler=train_sampler,
        collate_fn=yolo_dataset_collate, worker_init_fn=worker_init_fn
    )
    gen_val = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size,
        num_workers=4, pin_memory=True, sampler=val_sampler,
        collate_fn=yolo_dataset_collate, worker_init_fn=worker_init_fn
    )

    #-------------------------------#
    # 蒸馏训练器初始化
    #-------------------------------#
    if use_distillation:
        trainer = LightKDTrainer(
            model=model_train,
            teacher_model=teacher_model,
            kd_loss=kd_loss,
            lambda_kd=distill_cfg.lambda_kd
        )
    else:
        trainer = None

    #-------------------------------#
    # 开始训练
    #-------------------------------#
    print('开始训练...')
    scaler = GradScaler() if fp16 else None
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 学习率更新
        set_optimizer_lr(optimizer, lr_scheduler_func(epoch))

        # 分布式训练采样器
        if distributed:
            train_sampler.set_epoch(epoch)

        # 冻结/解冻切换
        if epoch >= Freeze_Epoch and Freeze_Train:
            batch_size = Unfreeze_batch_size
            # 重新计算学习率
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 0.1), lr_limit_max * 0.1)
            set_optimizer_lr(optimizer, Init_lr_fit)
            # 解冻backbone
            for param in model.backbone.parameters():
                param.requires_grad = True
            Freeze_Train = False

        # 训练一个epoch
        if use_distillation:
            # 蒸馏训练
            train_loss = trainer.train_one_epoch(
                epoch=epoch,
                train_loader=gen,
                optimizer=optimizer,
                device=device,
                scaler=scaler
            )
        else:
            # 常规训练（调用原有fit_one_epoch）
            train_loss = fit_one_epoch(
                model_train, model, yolo_loss, ema, optimizer,
                epoch, gen, gen_val, num_train, num_val,
                UnFreeze_Epoch, device, fp16, scaler,
                loss_history, eval_callback, log_dir, Cuda
            )

        # 保存模型和日志
        loss_history.append_loss(epoch, train_loss['total_loss'], 0)  # 简化示例
        if (epoch + 1) % save_period == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f"ep{epoch+1:03d}-loss{train_loss['total_loss']:.3f}.pth"))

    # 训练结束保存最终模型
    torch.save(model.state_dict(), os.path.join(log_dir, "last_epoch.pth"))
    print('训练完成!')