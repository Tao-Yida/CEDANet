import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def get_dataset_name_from_path(dataset_path):
    """
    从数据集路径中提取数据集名称
    Args:
        dataset_path: 数据集路径，如 'data/ijmond_data/train'
    Returns:
        str: 数据集名称，如 'ijmond_data_train'
    """
    # 移除末尾的斜杠并规范化路径
    path = os.path.normpath(dataset_path.rstrip("/"))
    path_parts = path.split(os.sep)

    # 移除常见的无意义部分
    filtered_parts = []
    skip_words = ["data", "dataset", "datasets"]

    for part in path_parts:
        if part.lower() not in skip_words and part.strip():
            filtered_parts.append(part)

    # 如果过滤后没有剩余部分，使用原始路径的最后两部分
    if not filtered_parts:
        filtered_parts = path_parts[-2:] if len(path_parts) >= 2 else path_parts[-1:]

    # 构建数据集名称
    dataset_name = "_".join(filtered_parts)

    # 清理名称，只保留字母、数字、下划线和连字符
    dataset_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in dataset_name)

    return dataset_name


def get_train_val_loaders(
    image_root,
    gt_root,
    trans_map_root,
    batchsize,
    trainsize,
    val_split=0.2,
    aug=True,
    freeze=False,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
    random_seed=42,
):
    """
    创建训练和校验数据加载器
    Args:
        image_root: 图像根目录
        gt_root: 真实标签根目录
        trans_map_root: 传输图根目录
        batchsize: 批量大小
        trainsize: 训练图像尺寸
        val_split: 校验集比例，默认0.2(20%)
        aug: 是否对训练集启用数据增强，默认True
        freeze: 是否冻结所有随机性以确保完全可重现，默认False
        shuffle: 是否打乱训练数据（当freeze=True时会被覆盖为False）
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存
        random_seed: 随机种子
    Returns:
        tuple: (train_loader, val_loader)
    """
    # freeze模式：设置所有随机种子以确保完全可重现
    if freeze:
        import torch
        import numpy as np
        import random

        # 设置所有随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 强制关闭shuffle以确保完全可重现
        shuffle = False
        print(f"[FREEZE MODE] 所有随机性已冻结，种子={random_seed}，shuffle已禁用")

    # 获取所有文件路径
    images = [image_root + f for f in os.listdir(image_root) if f.endswith((".jpg", ".png"))]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith((".jpg", ".png"))]
    trans = [trans_map_root + f for f in os.listdir(trans_map_root) if f.endswith((".jpg", ".png"))]

    # 排序确保一致性
    images = sorted(images)
    gts = sorted(gts)
    trans = sorted(trans)

    # 创建索引列表
    indices = list(range(len(images)))

    # 划分训练集和校验集索引
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed, shuffle=True)

    # 创建数据集
    train_dataset = SalObjDataset_from_indices(images, gts, trans, train_indices, trainsize, aug=aug, freeze=freeze)
    val_dataset = SalObjDataset_from_indices(images, gts, trans, val_indices, trainsize, aug=False, freeze=freeze)  # 验证时不使用数据增强

    print(f"Total dataset size: {len(images)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


class SalObjDataset_from_indices(data.Dataset):
    """基于索引的数据集类，用于训练/校验分割"""

    def __init__(self, images, gts, trans, indices, trainsize, aug=False, freeze=False):
        self.trainsize = trainsize
        self.aug = aug
        self.freeze = freeze
        self.images = [images[i] for i in indices]
        self.gts = [gts[i] for i in indices]
        self.trans = [trans[i] for i in indices]

        # 过滤不匹配的文件
        self.filter_files()
        self.size = len(self.images)

        # 数据变换 - 根据aug和freeze参数决定是否使用数据增强
        if self.aug and not self.freeze:
            # 训练时使用数据增强 - 只使用颜色增强以避免复杂的同步问题
            self.img_color_transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),  # 颜色扰动
                ]
            )
        else:
            # freeze模式下禁用所有随机变换，或者不使用增强
            self.img_color_transform = None
            if self.freeze and self.aug:
                print(f"[FREEZE MODE] 数据增强已禁用以确保完全可重现")

        # 基础变换（总是应用）
        self.img_basic_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # GT和trans的基础变换
        self.gt_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
        self.trans_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        tran = self.binary_loader(self.trans[index])

        # 应用颜色变换（仅对图像，在转换为tensor之前）
        if self.img_color_transform is not None:
            image = self.img_color_transform(image)

        # 应用基础变换
        image = self.img_basic_transform(image)
        gt = self.gt_transform(gt)
        tran = self.trans_transform(tran)

        return image, gt, tran

    def filter_files(self):
        """过滤尺寸不匹配的文件"""
        assert len(self.images) == len(self.gts)
        assert len(self.images) == len(self.trans)

        images = []
        gts = []
        trans = []

        for img_path, gt_path, tran_path in zip(self.images, self.gts, self.trans):
            try:
                img = Image.open(img_path)
                gt = Image.open(gt_path)
                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
                    trans.append(tran_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        self.images = images
        self.gts = gts
        self.trans = trans

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def __len__(self):
        return self.size
