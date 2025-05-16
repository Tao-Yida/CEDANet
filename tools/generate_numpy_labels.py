# save as: generate_numpy_labels.py
import os
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mutils


def is_image_file(filename):
    """判断文件是否为常见图像文件。"""
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return filename.lower().endswith(IMG_EXTS)


def json_to_labelmat(coco, img_id, h, w):
    """
    根据COCO标注和图片尺寸生成标签矩阵。
    只处理category_id为1或2的标注。
    """
    label = np.zeros((h, w), dtype=np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
        cid = ann["category_id"]
        if cid not in (1, 2):  # 只处理类别1和2
            continue
        rle = coco.annToRLE(ann)
        mask = mutils.decode(rle).astype(bool)
        label[mask] = cid
    return label


def generate_all_masks(img_dir, ann_file, out_dir="masks"):
    """
    读取COCO标注文件和图像文件夹，为每张图片生成对应的npy标签掩码。
    只处理图像文件夹中实际存在的图片（且为图像文件）。
    """
    os.makedirs(out_dir, exist_ok=True)
    # 只保留图像文件
    img_files = set(f for f in os.listdir(img_dir) if is_image_file(f))
    coco = COCO(ann_file)
    for img_info in tqdm(coco.loadImgs(coco.getImgIds()), desc="Generating masks"):
        file_name = img_info["file_name"]
        if file_name not in img_files:
            print(f"Warning: {file_name} not found in {img_dir}, skipping.")
            continue  # 跳过未找到的图片
        h, w = img_info["height"], img_info["width"]
        img_id = img_info["id"]
        label = json_to_labelmat(coco, img_id, h, w)
        out_name = os.path.splitext(file_name)[0] + ".npy"
        np.save(os.path.join(out_dir, out_name), label)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        default="smoke-segmentation.v5i.coco-segmentation/test",
        help="Path to image folder",
    )
    parser.add_argument(
        "--ann",
        default="smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json",
        help="Path to COCO annotation file",
    )
    parser.add_argument(
        "--out",
        default="smoke-segmentation.v5i.coco-segmentation/masks",
        help="Output folder for .npy masks",
    )
    args = parser.parse_args()
    generate_all_masks(args.img_dir, args.ann, args.out)
