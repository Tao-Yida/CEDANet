# save as: dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import json


class SmokeDataset_Seg(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        """
        :param img_dir: 图像文件夹路径
        :param mask_dir: 掩码文件夹路径
        :param transform: 图像变换
        :param target_transform: 掩码变换
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        # 获取所有图像文件名（不含扩展名），并确保对应的mask存在
        self.imgs = []
        all_img_files = [
            f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        for img_file in all_img_files:
            base_name = os.path.splitext(img_file)[0]
            mask_path = os.path.join(mask_dir, base_name + ".npy")
            if os.path.exists(mask_path):
                self.imgs.append(base_name)
            else:
                print(f"Warning: Mask not found for image {img_file}, skipping.")

        if not self.imgs:
            raise RuntimeError(f"No image/mask pairs found in {img_dir} and {mask_dir}")
        print(f"Found {len(self.imgs)} image/mask pairs.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        base_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, base_name + ".jpg")  # 假设都是jpg，如果不是需要更灵活处理
        mask_path = os.path.join(self.mask_dir, base_name + ".npy")

        try:
            image = Image.open(img_path).convert("RGB")
            mask = np.load(mask_path)  # mask已经是 (H, W) 的 numpy 数组
        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            # 返回一个占位符或引发异常，取决于你的错误处理策略
            # 这里我们简单地重新抛出异常
            raise e
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            raise e

        # --- 应用变换 ---
        # 注意：对图像和掩码应用相同的几何变换（如Resize, RandomCrop）很重要
        # 但对图像应用颜色变换（如Normalize），对掩码不应用

        # 暂存随机状态，确保图像和掩码应用相同的随机几何变换
        # state = torch.get_rng_state()
        if self.transform:
            image = self.transform(image)
        # torch.set_rng_state(state) # 恢复状态，但 torchvision transform 通常不直接支持这种方式
        # 更好的方法是自定义 transform 或使用支持同步变换的库如 albumentations

        if self.target_transform:
            # PIL Image is needed for some transforms like Resize
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = self.target_transform(mask_pil)
            # Convert back to tensor, ensure it's LongTensor for CrossEntropyLoss
            mask = torch.from_numpy(np.array(mask_pil)).long()
        else:
            # Default: Convert numpy mask directly to LongTensor
            mask = torch.from_numpy(mask).long()

        # 确保掩码没有通道维度，形状为 (H, W)
        mask = mask.squeeze()

        return image, mask


class SmokeDataset_Img(Dataset):
    """
    用于图像级别分类的Dataset，从json文件中读取图像和对应的标签
    """

    @staticmethod  # 改为静态方法，不需要传递类实例
    def _to_list(obj):
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        return [obj]

    def __init__(self, img_dir, label_path, class_map_path, transform=None, target_transform=None):
        """
        :param img_dir: 图像文件夹路径
        :param label_path: json文件路径，包含图像和标签的映射
        :param class_map_path: 类别映射json文件路径
        :param transform: 图像变换
        :param target_transform: 标签变换
        """
        self.img_dir = img_dir
        self.label_path = label_path
        self.transform = transform
        self.target_transform = target_transform
        # load class mapping
        self.cls2id = self.load_class_map(class_map_path)
        # read image-level annotations
        records = json.load(open(self.label_path, "r", encoding="utf-8"))

        # a) 统计多标签
        multi_label_count = 0
        for item in records:
            ids = self._to_list(item.get("label_id"))
            labels = self._to_list(item.get("labels"))
            if len(ids) > 1 or len(labels) > 1:
                multi_label_count += 1
                # print(f"多标签图像: {item.get('file_name')}, " f"labels={labels}, label_ids={ids}")
        if multi_label_count:
            print(f"发现 {multi_label_count} 个多标签图像")

        # b) 构建 label_map
        label_map = {}
        for item in records:
            fn = item.get("file_name")
            ids = self._to_list(item.get("label_id"))
            labs = self._to_list(item.get("labels"))

            if fn is None:
                continue

            # 对于多标签分类:
            if ids:
                label_map[fn] = [int(id) for id in ids]  # 存储所有标签
            elif labs:
                label_map[fn] = [self.cls2id[lab] for lab in labs if lab in self.cls2id]  # 存储所有有效标签

        # collect valid images
        self.imgs = []
        self.labels = []
        self.filenames = []  # 保存原始文件名用于调试
        self.image_ids = []  # 新增：保存原始image_id
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        for img_file in os.listdir(self.img_dir):
            if not img_file.lower().endswith(exts):
                continue
            if img_file not in label_map:
                print(f"Warning: Annotation not found for {img_file}, skipping.")
                continue
            self.imgs.append(img_file)
            self.labels.append(label_map[img_file])
            self.filenames.append(img_file)

            # 获取对应的image_id
            for item in records:
                if item.get("file_name") == img_file:
                    self.image_ids.append(item.get("image_id"))
                    break

        if not self.imgs:
            raise RuntimeError(f"No images with annotations found in {self.img_dir}")
        print(f"加载了 {len(self.imgs)} 个有效样本")

    def __len__(self):
        return len(self.imgs)

    def load_class_map(self, path):
        """
        从json文件中读取类别映射
        :param path: json文件路径
        :return: 类别映射字典，键为类别名称，值为对应的整数标签
        """
        raw_json = json.load(open(path, "r", encoding="utf-8"))
        cls2id = {v: int(k) for k, v in raw_json.items()}
        return cls2id

    def __getitem__(self, idx):
        img_file = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_ids = self.labels[idx]
        # 创建多热编码向量 [0,0,1,0,1,0] 表示类别2和4被激活
        multi_hot = torch.zeros(len(self.cls2id))
        for label_id in label_ids:
            multi_hot[label_id] = 1
        # 返回 (图像, 标签) 以及文件名，方便调试
        return image, multi_hot, self.filenames[idx]


if __name__ == "__main__":
    import argparse
    import torchvision.transforms as T
    import random

    parser = argparse.ArgumentParser(description="Test SmokeDataset_Img")
    parser.add_argument(
        "--img_dir",
        type=str,
        default="smoke-segmentation.v5i.coco-segmentation/test",
        help="图像目录路径，默认 smoke-segmentation.v5i.coco-segmentation/test",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="smoke-segmentation.v5i.coco-segmentation/image_level_labels.json",
        help="图像级标签 JSON 文件路径，默认 smoke-segmentation.v5i.coco-segmentation/image_level_labels.json",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default="smoke-segmentation.v5i.coco-segmentation/class_map.json",
        help="类别映射 JSON 文件路径，默认 smoke-segmentation.v5i.coco-segmentation/class_map.json",
    )
    args = parser.parse_args()

    # 简单转换：将 PIL 转 Tensor
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    ds = SmokeDataset_Img(args.img_dir, args.label_path, args.class_map, transform=transform)
    print(f"Total samples: {len(ds)}")

    # 尝试读取ID为232的样本（有多标签）
    if len(ds) > 232:
        img, label, filename = ds[232]
        print(f"Sample[232] filename: {filename}")
        print(f"Sample[232] shape: {img.shape}, label: {label}")

    # 随机抽取十个样本
    sample_count = min(10, len(ds))
    indices = random.sample(range(len(ds)), sample_count)
    for idx in indices:
        img, label, filename = ds[idx]  # 接收所有3个返回值
        fn = ds.filenames[idx]
        orig_id = ds.image_ids[idx]
        print(f"Sample[{idx}] file: {fn}, original_id: {orig_id}, shape: {img.shape}, label: {label}")
