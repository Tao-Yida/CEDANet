import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trans_map_root, trainsize, aug):
        self.trainsize = trainsize
        self.aug = aug
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith(".jpg") or f.endswith(".png")]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith(".jpg") or f.endswith(".png")]
        self.trans = [trans_map_root + f for f in os.listdir(trans_map_root) if f.endswith(".jpg") or f.endswith(".png")]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.trans = sorted(self.trans)
        self.filter_files()
        self.size = len(self.images)
        if self.aug:
            self.img_transform = transforms.Compose(
                [
                    transforms.RandomRotation(15),
                    # transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
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
        gt = self.binary_loader(self.gts[index])  # 0 or 255
        tran = self.binary_loader(self.trans[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        tran = self.trans_transform(tran)
        return image, gt, tran

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        assert len(self.images) == len(self.trans)
        images = []
        gts = []
        trans = []
        for img_path, gt_path, tran_path in zip(self.images, self.gts, self.trans):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                trans.append(tran_path)
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
            # return img.convert('1')
            return img.convert("L")

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class test_dataset(data.Dataset):
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

    def grayscale_loader(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img


def get_loader(image_root, gt_root, trans_map_root, batchsize, trainsize, aug=False, shuffle=True, num_workers=8, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, trans_map_root, trainsize, aug)
    print(f"Length of dataset: {len(dataset)}")
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader
