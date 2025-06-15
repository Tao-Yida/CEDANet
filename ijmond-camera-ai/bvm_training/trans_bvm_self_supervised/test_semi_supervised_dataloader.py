#!/usr/bin/env python3
"""
测试半监督数据加载器的基本功能
"""

import sys
import os

sys.path.append("/home/ytao/Thesis/ijmond-camera-ai/bvm_training/trans_bvm_self_supervised")

from dataloader import get_loader, get_train_val_loaders


def test_get_loader():
    """测试 get_loader 函数"""
    print("=== 测试 get_loader 函数 ===")

    # 使用现有的数据集路径进行测试
    dataset_path = "/home/ytao/Thesis/data/ijmond_data/test"
    image_root = os.path.join(dataset_path, "img/")
    gt_root = os.path.join(dataset_path, "gt/")
    trans_map_root = os.path.join(dataset_path, "trans/")

    # 检查路径是否存在
    if not all(os.path.exists(path) for path in [image_root, gt_root, trans_map_root]):
        print(f"跳过测试: 数据路径不存在")
        print(f"  - 图像路径: {image_root} ({'存在' if os.path.exists(image_root) else '不存在'})")
        print(f"  - GT路径: {gt_root} ({'存在' if os.path.exists(gt_root) else '不存在'})")
        print(f"  - Trans路径: {trans_map_root} ({'存在' if os.path.exists(trans_map_root) else '不存在'})")
        return

    try:
        # 测试不同配置
        configs = [
            {"aug": False, "freeze": False, "name": "标准配置（无数据增强）"},
            {"aug": True, "freeze": False, "name": "启用数据增强"},
            {"aug": True, "freeze": True, "name": "冻结模式（数据增强将被禁用）"},
        ]

        for config in configs:
            print(f"\n--- 测试配置: {config['name']} ---")
            loader = get_loader(
                image_root=image_root,
                gt_root=gt_root,
                trans_map_root=trans_map_root,
                batchsize=2,
                trainsize=352,
                aug=config["aug"],
                freeze=config["freeze"],
                shuffle=True,
                num_workers=0,  # 避免多进程问题
            )

            print(f"数据加载器创建成功!")
            print(f"批次数量: {len(loader)}")

            # 测试加载一个批次
            for i, (images, gts, trans) in enumerate(loader):
                print(f"批次 {i+1}: 图像形状={images.shape}, GT形状={gts.shape}, Trans形状={trans.shape}")
                if i >= 0:  # 只测试第一个批次
                    break

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_get_train_val_loaders():
    """测试 get_train_val_loaders 函数"""
    print("\n=== 测试 get_train_val_loaders 函数 ===")

    # 使用现有的数据集路径进行测试
    dataset_path = "/home/ytao/Thesis/data/ijmond_data/test"
    image_root = os.path.join(dataset_path, "img/")
    gt_root = os.path.join(dataset_path, "gt/")
    trans_map_root = os.path.join(dataset_path, "trans/")

    # 检查路径是否存在
    if not all(os.path.exists(path) for path in [image_root, gt_root, trans_map_root]):
        print(f"跳过测试: 数据路径不存在")
        return

    try:
        print(f"\n--- 测试训练/验证分割 ---")
        train_loader, val_loader = get_train_val_loaders(
            image_root=image_root,
            gt_root=gt_root,
            trans_map_root=trans_map_root,
            batchsize=2,
            trainsize=352,
            val_split=0.2,
            aug=True,
            freeze=False,
            shuffle=True,
            num_workers=0,
        )

        print(f"训练/验证加载器创建成功!")
        print(f"训练批次数量: {len(train_loader)}")
        print(f"验证批次数量: {len(val_loader)}")

        # 测试加载一个批次
        for i, (images, gts, trans) in enumerate(train_loader):
            print(f"训练批次 {i+1}: 图像形状={images.shape}")
            break

        for i, (images, gts, trans) in enumerate(val_loader):
            print(f"验证批次 {i+1}: 图像形状={images.shape}")
            break

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_get_loader()
    test_get_train_val_loaders()
    print("\n=== 测试完成 ===")
