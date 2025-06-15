import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import random


def _setup_reproducibility(freeze, random_seed, verbose=True):
    """
    统一的随机性控制函数
    Args:
        freeze: 是否冻结所有随机性
        random_seed: 随机种子
        verbose: 是否打印信息
    Returns:
        bool: 实际的shuffle设置
    """
    if freeze:
        # 设置所有随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if verbose:
            print(f"[FREEZE MODE] 所有随机性已冻结，种子={random_seed}，shuffle已禁用")
        return False  # 返回shuffle=False
    return True  # 返回shuffle=True


def _get_file_paths(image_root, gt_root, trans_map_root):
    """
    获取所有文件路径并排序
    Args:
        image_root: 图像根目录
        gt_root: 真实标签根目录
        trans_map_root: 传输图根目录
    Returns:
        tuple: (images, gts, trans) 路径列表
    """
    images = [image_root + f for f in os.listdir(image_root) if f.endswith((".jpg", ".png"))]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith((".jpg", ".png"))]
    trans = [trans_map_root + f for f in os.listdir(trans_map_root) if f.endswith((".jpg", ".png"))]

    # 排序确保一致性
    return sorted(images), sorted(gts), sorted(trans)


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
    # 统一处理随机性设置
    actual_shuffle = _setup_reproducibility(freeze, random_seed) and shuffle

    # 获取文件路径
    images, gts, trans = _get_file_paths(image_root, gt_root, trans_map_root)

    # 创建索引列表并划分训练集和校验集
    indices = list(range(len(images)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed, shuffle=True)

    # 创建数据集
    train_dataset = SalObjDataset.from_indices(images, gts, trans, train_indices, trainsize, aug=aug, freeze=freeze)
    val_dataset = SalObjDataset.from_indices(images, gts, trans, val_indices, trainsize, aug=False, freeze=freeze)  # 验证时不使用数据增强

    print(f"Total dataset size: {len(images)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=batchsize, shuffle=actual_shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


class SalObjDataset(data.Dataset):
    """统一的显著性对象数据集类"""

    def __init__(self, all_images, all_gts, all_trans, trainsize, indices=None, aug=False, freeze=False):
        """
        Args:
            all_images: 所有图像路径列表
            all_gts: 所有GT路径列表
            all_trans: 所有传输图路径列表
            trainsize: 训练图像尺寸
            indices: 索引列表，如果为None则使用所有数据
            aug: 是否启用数据增强
            freeze: 是否冻结随机性
        """
        self.trainsize = trainsize
        self.aug = aug
        self.freeze = freeze

        # 根据索引选择数据
        if indices is not None:
            self.images = [all_images[i] for i in indices]
            self.gts = [all_gts[i] for i in indices]
            self.trans = [all_trans[i] for i in indices]
        else:
            self.images = all_images
            self.gts = all_gts
            self.trans = all_trans

        # 过滤不匹配的文件
        self._filter_files()
        self.size = len(self.images)

        # 初始化数据变换
        self._setup_transforms()

    @classmethod
    def from_indices(cls, all_images, all_gts, all_trans, indices, trainsize, aug=False, freeze=False):
        """
        类方法：从索引创建数据集实例
        """
        return cls(all_images, all_gts, all_trans, trainsize, indices=indices, aug=aug, freeze=freeze)

    def _setup_transforms(self):
        """设置数据变换"""
        # 颜色增强（只在需要时应用）
        if self.aug and not self.freeze:
            self.img_color_transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                ]
            )
        else:
            self.img_color_transform = None
            if self.freeze and self.aug:
                print("[FREEZE MODE] 数据增强已禁用以确保完全可重现")

        # 基础变换（总是应用）
        self.img_basic_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # GT和trans的变换
        self.gt_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
        self.trans_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):
        """获取数据项"""
        image = self._rgb_loader(self.images[index])
        gt = self._binary_loader(self.gts[index])
        trans = self._binary_loader(self.trans[index])

        # 应用颜色变换（仅对图像）
        if self.img_color_transform is not None:
            image = self.img_color_transform(image)

        # 应用基础变换
        image = self.img_basic_transform(image)
        gt = self.gt_transform(gt)
        trans = self.trans_transform(trans)

        return image, gt, trans

    def _filter_files(self):
        """过滤尺寸不匹配的文件"""
        assert len(self.images) == len(self.gts) == len(self.trans), "图像、GT和传输图数量不匹配"

        filtered_images, filtered_gts, filtered_trans = [], [], []

        for img_path, gt_path, trans_path in zip(self.images, self.gts, self.trans):
            try:
                with Image.open(img_path) as img, Image.open(gt_path) as gt:
                    if img.size == gt.size:
                        filtered_images.append(img_path)
                        filtered_gts.append(gt_path)
                        filtered_trans.append(trans_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        self.images = filtered_images
        self.gts = filtered_gts
        self.trans = filtered_trans

    def _rgb_loader(self, path):
        """加载RGB图像"""
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def _binary_loader(self, path):
        """加载二值图像"""
        with open(path, "rb") as f:
            return Image.open(f).convert("L")

    def __len__(self):
        return self.size


class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith(".jpg") or f.endswith(".png")]
        self.images = sorted(self.images)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.size = len(self.images)
        self.index = 0

    def load_gt(self, name):
        if os.path.exists(name + ".jpg"):
            image = self.binary_loader(name + ".jpg")
        else:
            image = self.binary_loader(name + ".png")
        return image

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split("/")[-1]
        if name.endswith(".jpg"):
            name = name.split(".jpg")[0] + ".png"
        self.index += 1
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")
