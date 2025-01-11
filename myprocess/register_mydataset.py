# import os
# from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets import load_coco_json
# from detectron2.data.datasets import register_coco_instances

# # 注册CoNSep数据集
# def register_dataset(dataset_folder, split):
#     json_file = os.path.join(dataset_folder, f"{split}_annotations.json")  # JSON文件路径
#     image_root = os.path.join(dataset_folder, f"{split.capitalize()}/image_crop")      # 图像文件夹路径
#     # 确保 JSON 文件和图像路径存在
#     if not os.path.exists(json_file):
#         print(f"Error: {json_file} not found.")
#         return

#     if not os.path.exists(image_root):
#         print(f"Error: {image_root} not found.")
#         return

#     DatasetCatalog.register(
#         f"consep_{split}",
#         lambda: load_coco_json(
#             json_file = json_file,
#             image_root = image_root,
#             dataset_name=f"consep_{split}"
#         )
#     )
#     # 设置Metadata
#     metadata = MetadataCatalog.get(f"consep_{split}")
#     metadata.set(thing_classes=["others", "epithelial", "spindle-shaped"],ignore_label = 0)

# dataset_folder = '/data/wxx/eclipse/CoNSeP'
# register_dataset(dataset_folder, "train")
# register_dataset(dataset_folder, "test")
# print("1111111111111111111111111111111111111111111111111111111111")
# print("注册数据集")
# print("1111111111111111111111111111111111111111111111111111111111")
# if 'consep_train' in DatasetCatalog.list():
#     print("Successfully registered!")
# else:
#     print("Fail to register!")

from detectron2.data import DatasetCatalog, MetadataCatalog
import os

def get_semantic_segmentation_dicts(img_dir, mask_dir):
    dataset_dicts = []
    for filename in os.listdir(img_dir):
        record = {}
        # Image file path
        record["file_name"] = os.path.join(img_dir, filename)
        
        # Corresponding mask file path (assuming '_mask' suffix in filename)
        record["sem_seg_file_name"] = os.path.join(mask_dir, filename)
        
        # Optionally, add height and width if they’re known
        record["height"], record["width"] = 320, 320  # Replace with your actual dimensions
        dataset_dicts.append(record)
    return dataset_dicts

dataset_dir = "/data/wxx/eclipse/CoNSeP"
# Paths to train image and mask directories
img_dir = os.path.join(dataset_dir,"Train/image_crop")
mask_dir = os.path.join(dataset_dir,"Train/mask_crop")

# Register the train dataset
DatasetCatalog.register("consep_train", lambda: get_semantic_segmentation_dicts(img_dir, mask_dir))
MetadataCatalog.get("consep_train").set(
        stuff_classes=["background","others", "epithelial", "spindle-shaped"],
        ignore_label = 0,
        evaluator_type = "sem_seg"
    )

# Paths to test image and mask directories
img_dir = os.path.join(dataset_dir,"Test/image_crop")
mask_dir = os.path.join(dataset_dir,"Test/mask_crop")

# Register the test dataset
DatasetCatalog.register("consep_test", lambda: get_semantic_segmentation_dicts(img_dir, mask_dir))
MetadataCatalog.get("consep_test").set(
        stuff_classes=["background","others", "epithelial", "spindle-shaped"],
        ignore_label = 0,
        evaluator_type = "sem_seg"
    )