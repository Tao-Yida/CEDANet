#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path


def find_and_copy_images(image_name):
    """
    从指定的9个文件夹中查找图片文件并复制到temp文件夹

    Args:
        image_name: 图片名称（不含扩展名），如 'img328_crop'
    """
    # 定义基础路径
    base_path = Path("/home/ytao/Thesis")

    # 定义8个源文件夹路径
    source_folders = [
        # 全图
        "data/ijmond_data/test/img",
        # 3类分割
        "data/ijmond_data/test/gt_3cls",
        # 全监督E1
        "results/supervised/ijmond/full-supervision/SMOKE5K-supervised/SMOKE5K_Dataset_SMOKE5K_train/SMOKE5K_Dataset_SMOKE5K_train_best_model",
        # 半监督E2
        "results/self_supervised/ijmond/SMOKE5K_Dataset_SMOKE5K_train_ssl_ijmond_camera_SMOKE5K-full_citizen_constraint/SMOKE5K_Dataset_SMOKE5K_train_ssl_ijmond_camera_SMOKE5K-full_citizen_constraint_epoch_020_from_scratch",
        # 半监督E3
        "results/thesis/ijmond/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_camera_SMOKE5K-full_non_constraint_da/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_camera_SMOKE5K-full_non_constraint_da_best_model",
        # 半监督E4
        "results/thesis/ijmond/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_camera_SMOKE5K-full_citizen_constraint_da/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_camera_SMOKE5K-full_citizen_constraint_da_best_model",
        # 半监督E5
        "results/thesis/ijmond/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_camera_SMOKE5K-full_expert_constraint_da/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_camera_SMOKE5K-full_expert_constraint_da_best_model",
        # 半监督E6
        "results/thesis/ijmond/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_data_train_da/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_data_train_da_best_model",
    ]

    # 目标文件夹
    target_folder = base_path / "temp"

    # 确保temp文件夹存在
    target_folder.mkdir(exist_ok=True)

    # 支持的图片格式
    image_extensions = [".png", ".jpg", ".jpeg"]

    print(f"正在查找图片: {image_name}")
    print(f"目标文件夹: {target_folder}")
    print("-" * 50)

    found_count = 0

    # 遍历每个源文件夹
    for i, folder_path in enumerate(source_folders, 1):
        full_folder_path = base_path / folder_path

        if not full_folder_path.exists():
            print(f"E{i}: 文件夹不存在 - {full_folder_path}")
            continue

        # 查找匹配的图片文件
        found_file = None
        for ext in image_extensions:
            potential_file = full_folder_path / f"{image_name}{ext}"
            if potential_file.exists():
                found_file = potential_file
                break

        if found_file:
            # 生成目标文件名
            file_extension = found_file.suffix
            target_filename = f"{image_name}_E{i}{file_extension}"
            target_file_path = target_folder / target_filename

            try:
                # 复制文件
                shutil.copy2(found_file, target_file_path)
                print(f"E{i}: 复制成功 - {found_file.name} -> {target_filename}")
                found_count += 1
            except Exception as e:
                print(f"E{i}: 复制失败 - {e}")
        else:
            print(f"E{i}: 未找到文件 - {folder_path}")

    print("-" * 50)
    print(f"总共找到并复制了 {found_count} 个文件")


def main():

    image_name = "img328_crop3"
    find_and_copy_images(image_name)


if __name__ == "__main__":
    main()
