"""
将COCO格式的标注文件转换为图像级别的标签。
此脚本读取COCO格式的标注文件，提取每张图像包含的目标类别，
并生成一个新的JSON文件，其中包含每张图像的文件名和对应的标签。
"""

import json

# COCO类别id到名称的映射
CATEGORY_ID_TO_NAME = {0: "smoke", 1: "high-opacity-smoke", 2: "low-opacity-smoke"}

# 读取COCO标注文件
with open(
    "/home/ytao/Thesis/smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json", "r", encoding="utf-8"
) as f:
    coco = json.load(f)  # type: dict

# 构建image_id到file_name的映射
image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

# 构建image_id到包含的类别集合的映射
image_id_to_categories = {img["id"]: set() for img in coco["images"]}
for ann in coco["annotations"]:
    image_id_to_categories[ann["image_id"]].add(ann["category_id"])

# 生成输出
output = []  # type: list
for image_id, file_name in image_id_to_name.items():
    categories = image_id_to_categories[image_id]
    if categories:
        labels = [CATEGORY_ID_TO_NAME[cid] for cid in sorted(categories)]
        label_ids = [cid for cid in sorted(categories)]
        output.append({"image_id": image_id, "file_name": file_name, "labels": labels, "label_id": label_ids})
    else:
        output.append({"image_id": image_id, "file_name": file_name, "labels": ["no-smoke"], "label_id": [0]})

with open(
    "/home/ytao/Thesis/smoke-segmentation.v5i.coco-segmentation/image_level_labels.json", "w", encoding="utf-8"
) as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
