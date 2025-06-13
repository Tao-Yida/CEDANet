import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
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
    image_root, gt_root, trans_map_root, batchsize, trainsize, val_split=0.2, shuffle=True, num_workers=12, pin_memory=True, random_seed=42
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
        shuffle: 是否打乱训练数据
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存
        random_seed: 随机种子
    Returns:
        tuple: (train_loader, val_loader)
    """
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
    train_dataset = SalObjDataset_from_indices(images, gts, trans, train_indices, trainsize)
    val_dataset = SalObjDataset_from_indices(images, gts, trans, val_indices, trainsize)

    print(f"Total dataset size: {len(images)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


# def get_loader(image_root, gt_root, trans_map_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True) -> data.DataLoader:
#     # 保持原有函数以确保向后兼容
#     dataset = SalObjDataset(image_root, gt_root, trans_map_root, trainsize)
#     print(f"Length of dataset: {len(dataset)}")
#     data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
#     return data_loader


class SalObjDataset_from_indices(data.Dataset):
    """基于索引的数据集类，用于训练/校验分割"""

    def __init__(self, images, gts, trans, indices, trainsize):
        self.trainsize = trainsize
        self.images = [images[i] for i in indices]
        self.gts = [gts[i] for i in indices]
        self.trans = [trans[i] for i in indices]

        # 过滤不匹配的文件
        self.filter_files()
        self.size = len(self.images)

        # 数据变换
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.gt_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
        self.trans_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        tran = self.binary_loader(self.trans[index])

        image = self.img_transform(image)
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


# class SalObjDataset(data.Dataset):
#     def __init__(self, image_root, gt_root, trans_map_root, trainsize):
#         self.trainsize = trainsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith(".jpg") or f.endswith(".png")]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith(".jpg") or f.endswith(".png")]
#         self.trans = [trans_map_root + f for f in os.listdir(trans_map_root) if f.endswith(".jpg") or f.endswith(".png")]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.trans = sorted(self.trans)
#         self.filter_files()
#         self.size = len(self.images)
#         self.img_transform = transforms.Compose(
#             [
#                 transforms.Resize((self.trainsize, self.trainsize)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#         self.gt_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
#         self.trans_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

#     def __getitem__(self, index):
#         image = self.rgb_loader(self.images[index])
#         gt = self.binary_loader(self.gts[index])
#         tran = self.binary_loader(self.trans[index])
#         image = self.img_transform(image)
#         gt = self.gt_transform(gt)
#         tran = self.trans_transform(tran)
#         return image, gt, tran

#     def filter_files(self):
#         assert len(self.images) == len(self.gts)
#         assert len(self.images) == len(self.trans)
#         images = []
#         gts = []
#         trans = []
#         for img_path, gt_path, tran_path in zip(self.images, self.gts, self.trans):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             if img.size == gt.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#                 trans.append(tran_path)
#         self.images = images
#         self.gts = gts
#         self.trans = trans

#     def rgb_loader(self, path):
#         with open(path, "rb") as f:
#             img = Image.open(f)
#             return img.convert("RGB")

#     def binary_loader(self, path):
#         with open(path, "rb") as f:
#             img = Image.open(f)
#             # return img.convert('1')
#             return img.convert("L")

#     def resize(self, img, gt):
#         assert img.size == gt.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.Resampling.BILINEAR), gt.resize((w, h), Image.Resampling.NEAREST)
#         else:
#             return img, gt

#     def __len__(self):
#         return self.size


# class test_dataset:
#     def __init__(self, image_root, testsize):
#         self.testsize = testsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith(".jpg") or f.endswith(".png")]
#         self.images = sorted(self.images)
#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize((self.testsize, self.testsize)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#         self.size = len(self.images)
#         self.index = 0

#     def load_gt(self, name):
#         if os.path.exists(name + ".jpg"):
#             image = self.binary_loader(name + ".jpg")
#         else:
#             image = self.binary_loader(name + ".png")
#         return image

#     def load_data(self):
#         image = self.rgb_loader(self.images[self.index])
#         HH = image.size[0]
#         WW = image.size[1]
#         image = self.transform(image)
#         image = image.unsqueeze(0)
#         name = self.images[self.index].split("/")[-1]
#         if name.endswith(".jpg"):
#             name = name.split(".jpg")[0] + ".png"
#         self.index += 1
#         return image, HH, WW, name

#     def rgb_loader(self, path):
#         with open(path, "rb") as f:
#             img = Image.open(f)
#             return img.convert("RGB")

#     def binary_loader(self, path):
#         with open(path, "rb") as f:
#             img = Image.open(f)
#             return img.convert("L")
