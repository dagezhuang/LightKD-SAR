import os
import numpy as np  # 新增：处理YOLO格式坐标转换

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    说明：
    - 针对sardet100k数据集（YOLO格式标注）修改，主要调整真实框读取逻辑
    - 保留原mAP计算流程，仅适配数据格式和类别
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定运行模式（同原逻辑）
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   指向sardet100k的类别文件（需与训练时一致）
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/sardet100k_classes.txt'  # 关键修改：sardet类别文件
    #--------------------------------------------------------------------------------------#
    #   mAP计算的IoU阈值（根据需求调整）
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   预测相关参数（同原逻辑，不建议修改）
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    nms_iou         = 0.5
    score_threhold  = 0.5
    map_vis         = False  # 如需可视化预测框与真实框对比，设为True
    #-------------------------------------------------------#
    #   指向sardet100k数据集根目录（关键修改）
    #-------------------------------------------------------#
    SARDet_path     = 'SARDet-100K'  # 替换为实际路径
    #-------------------------------------------------------#
    #   结果输出文件夹
    #-------------------------------------------------------#
    map_out_path    = 'map_out_sardet'  # 建议单独文件夹，避免与其他结果混淆

    # 获取验证集图片ID（从val/images中提取，不含后缀）
    val_images_dir = os.path.join(SARDet_path, 'val', 'images')
    image_ids = [os.path.splitext(f)[0] for f in os.listdir(val_images_dir) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 创建输出目录
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    for subdir in ['ground-truth', 'detection-results', 'images-optional']:
        if not os.path.exists(os.path.join(map_out_path, subdir)):
            os.makedirs(os.path.join(map_out_path, subdir))

    class_names, _ = get_classes(classes_path)

    #------------------------------#
    #   1. 生成预测结果（同原逻辑）
    #------------------------------#
    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(val_images_dir, f"{image_id}.jpg")  # 假设图片为jpg格式
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, f"images-optional/{image_id}.jpg"))
            # 调用yolo的get_map_txt生成预测框txt
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    #------------------------------#
    #   2. 生成真实框（关键修改：适配YOLO格式）
    #------------------------------#
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        val_labels_dir = os.path.join(SARDet_path, 'val', 'labels')  # sardet验证集标注目录
        for image_id in tqdm(image_ids):
            label_path = os.path.join(val_labels_dir, f"{image_id}.txt")
            with open(os.path.join(map_out_path, f"ground-truth/{image_id}.txt"), "w") as new_f:
                if os.path.exists(label_path):
                    # 读取YOLO格式标注（class_id x_center y_center w h，归一化）
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    # 读取图片尺寸，用于坐标转换
                    image_path = os.path.join(val_images_dir, f"{image_id}.jpg")
                    with Image.open(image_path) as img:
                        img_w, img_h = img.size  # 图片宽、高
                    # 转换为VOC格式（class_name xmin ymin xmax ymax）
                    for line in lines:
                        line = line.strip().split()
                        if len(line) != 5:
                            continue  # 跳过无效行
                        class_id, x_center, y_center, w, h = map(float, line)
                        class_id = int(class_id)
                        # 检查类别ID是否在有效范围内
                        if class_id < 0 or class_id >= len(class_names):
                            continue
                        class_name = class_names[class_id]
                        # 归一化坐标转绝对坐标
                        xmin = (x_center - w/2) * img_w
                        ymin = (y_center - h/2) * img_h
                        xmax = (x_center + w/2) * img_w
                        ymax = (y_center + h/2) * img_h
                        # 写入真实框（保留整数坐标）
                        new_f.write(f"{class_name} {int(xmin)} {int(ymin)} {int(xmax)} {int(ymax)}\n")
        print("Get ground truth result done.")

    #------------------------------#
    #   3. 计算mAP（同原逻辑）
    #------------------------------#
    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    #------------------------------#
    #   4. COCO格式评估（如需）
    #------------------------------#
    if map_mode == 4:
        print("Get map (COCO format).")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")