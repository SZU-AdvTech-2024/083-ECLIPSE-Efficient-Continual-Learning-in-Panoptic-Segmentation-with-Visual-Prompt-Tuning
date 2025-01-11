import os
import json
import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.measure import find_contours

def create_coco_json(image_files, image_folder, mask_folder, json_output_path):
    # 定义类别映射
    category_mapping = {
        1: 'others',            # Category 1: others (inflammatory + other)
        2: 'epithelial',        # Category 2: epithelial (healthy + dysplastic/malignant)
        3: 'spindle-shaped'     # Category 3: spindle-shaped (fibroblast, muscle, endothelial)
    }

    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'others'},
            {'id': 2, 'name': 'epithelial'},
            {'id': 3, 'name': 'spindle-shaped'}
        ],
        'licenses': [],
        'info': {}
    }

    annotation_id = 1  # 初始化注释ID

    # 处理每个图像文件
    for image_file in image_files:
        # 去掉文件后缀 ".png"，只保留文件名
        image_name_without_extension = os.path.splitext(image_file)[0]
        
        # 读取图像
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        # 添加图像信息到COCO数据
        coco_data['images'].append({
            'id': image_name_without_extension,  # 使用去掉后缀的文件名作为图像ID
            'file_name': os.path.relpath(image_path, start=image_folder),
            'width': width,
            'height': height
        })
        
        # 读取对应的掩码图像
        mask_file = image_file
        mask_path = os.path.join(mask_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取掩码
        
        # 标记图像中的不同实例
        labeled_mask = label(mask)  # 使用skimage的label函数标记连通区域
        
        # 处理每个实例
        for region in regionprops(labeled_mask):
            minr, minc, maxr, maxc = region.bbox  # 获取边界框
            instance_mask = (labeled_mask == region.label).astype(np.uint8)  # 生成当前实例的二值掩码
            category_id = int(np.mean(mask[region.coords[:, 0], region.coords[:, 1]]))  # 计算该区域的平均像素值
            
            # 如果类别映射中没有此类别，跳过
            if category_id not in category_mapping:
                continue  # 忽略未定义类别的实例
            
            # 获取实例的分割轮廓
            segmentation = []
            # contours = find_contours(instance_mask, 0.5)
            # for contour in contours:
            #     # 转换为COCO格式需要的 x, y 坐标平铺列表
            #     contour = np.flip(contour, axis=1)  # 翻转为 [x, y] 顺序
            #     flattened = contour.ravel().tolist()  # 将坐标展平成一维列表
            #     if len(flattened) >= 6:  # 只有包含至少3个点的轮廓才有效
            #         segmentation.append([int(round(coord)) for coord in approx.flatten().tolist()])

            contours = find_contours(instance_mask, 0.5)  # 找到轮廓
            for contour in contours:
                contour = np.flip(contour, axis=1)  # 翻转坐标使之符合图像坐标系
                epsilon = 0.02 * cv2.arcLength(contour.astype(np.float32), True)  # epsilon为近似精度
                approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True)  # 获取近似多边形
                
                if len(approx) >= 3:  # 有效多边形
                    segmentation.append([int(round(coord)) for coord in approx.flatten().tolist()])

            if segmentation:  # 确保分割区域存在
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_name_without_extension,  # 图像ID使用去掉后缀的文件名
                    'category_id': category_id,
                    'segmentation': segmentation,  # 多边形分割区域
                    'area': region.area,  # 区域面积
                    'bbox': [minc, minr, maxc - minc, maxr - minr],  # 边界框
                    'iscrowd': 0  # 是否为crowd区域（此处设置为0，表示不是crowd）
                })
                annotation_id += 1
    
    # 保存COCO格式的JSON
    with open(json_output_path, 'w') as json_file:
        json.dump(coco_data, json_file)

# 数据集路径
dataset = '/data/wxx/eclipse/'

# 文件夹路径
train_image_folder = dataset + 'CoNSeP/Train/image_crop'
train_mask_folder = dataset + 'CoNSeP/Train/mask_crop'
test_image_folder = dataset + 'CoNSeP/Test/image_crop'
test_mask_folder = dataset + 'CoNSeP/Test/mask_crop'

# 图像文件列表
train_image_files = sorted(os.listdir(train_image_folder))
test_image_files = sorted(os.listdir(test_image_folder))

# 处理训练集并保存
train_json_output = dataset + 'CoNSeP/train_annotations.json'
create_coco_json(train_image_files, train_image_folder, train_mask_folder, train_json_output)

# 处理测试集并保存
test_json_output = dataset + 'CoNSeP/test_annotations.json'
create_coco_json(test_image_files, test_image_folder, test_mask_folder, test_json_output)
