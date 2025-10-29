#--------------------------------------------#
#   该部分代码用于查看网络结构和计算参数量/ FLOPs
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

# 根据代码库结构调整导入路径
from models.yolo_pafpn import YOLOPAFPN
from models.yolo_head import YOLOXHead
from nets.yolo import YoloBody  # 若使用集成模型类

if __name__ == "__main__":
    #-------------------------------#
    #   配置参数（根据实际模型修改）
    #-------------------------------#
    input_shape = [640, 640]  # 输入图像尺寸
    num_classes = 6  # SARDet数据集类别数（参考exp/default.py中的配置）
    depth = 0.33     # 模型深度因子（参考exp/default.py中的配置）
    width = 0.50     # 模型宽度因子（参考exp/default.py中的配置）
    phi = 's'        # 模型版本（与depth/width对应，如's'对应0.33/0.50）
    
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #-------------------------------#
    #   方式1：查看完整模型结构
    #-------------------------------#
    print("===== 完整模型结构 =====")
    model = YoloBody(num_classes, phi).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))  # 3表示RGB通道
    
    #-------------------------------#
    #   方式2：分别查看backbone和head（可选）
    #-------------------------------#
    print("\n===== Backbone (YOLOPAFPN) 结构 =====")
    backbone = YOLOPAFPN(depth, width).to(device)
    summary(backbone, (3, input_shape[0], input_shape[1]))
    
    print("\n===== Head (YOLOXHead) 结构 =====")
    # 输入通道数需与backbone输出匹配（参考models/yolo_pafpn.py的out_channels）
    head = YOLOXHead(num_classes, width, in_channels=[256, 512, 1024]).to(device)
    # 模拟backbone输出的特征图尺寸（640/8=80, 640/16=40, 640/32=20）
    dummy_feats = [
        torch.randn(1, 256, 80, 80).to(device),
        torch.randn(1, 512, 40, 40).to(device),
        torch.randn(1, 1024, 20, 20).to(device)
    ]
    summary(head, input_data=dummy_feats)  # 查看head结构
    
    #-------------------------------#
    #   计算参数量和FLOPs
    #-------------------------------#
    print("\n===== 计算参数量和FLOPs =====")
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    
    # 参考YOLOX论文，卷积操作计为2次运算（乘法+加法）
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f'Total GFLOPS: {flops}')
    print(f'Total params: {params}')