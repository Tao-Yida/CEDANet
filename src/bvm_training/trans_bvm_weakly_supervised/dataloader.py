import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cv2


def _setup_reproducibility(freeze, random_seed, verbose=True):
    """
    Unified randomness control function
    Args:
        freeze: Whether to freeze all randomness
        random_seed: Random seed
        verbose: Whether to print info
    Returns:
        bool: Actual shuffle setting
    """
    if freeze:
        # Set all random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if verbose:
            print(f"[FREEZE MODE] All randomness frozen, seed={random_seed}, shuffle disabled")
        return False
    return True


def _get_file_paths(image_root, gt_root, trans_map_root):
    """
    Get all file paths and sort them
    Args:
        image_root: Image root directory
        gt_root: Ground truth root directory
        trans_map_root: Transmission map root directory
    Returns:
        tuple: (images, gts, trans) path lists
    """
    images = [image_root + f for f in os.listdir(image_root) if f.endswith((".jpg", ".png"))]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith((".jpg", ".png"))]
    trans = [trans_map_root + f for f in os.listdir(trans_map_root) if f.endswith((".jpg", ".png"))]

    # Sort to ensure consistency
    return sorted(images), sorted(gts), sorted(trans)


def _print_augmentation_status(aug, freeze, dataset_type=""):
    """
    Print data augmentation status info
    Args:
        aug: Whether to enable data augmentation
        freeze: Whether in freeze mode
        dataset_type: Dataset type description, e.g. "labeled data" or "unlabeled data"
    """
    if aug and not freeze:
        status = "ENABLED"
    elif freeze:
        status = "DISABLED (freeze mode)"
    else:
        status = "DISABLED"

    if dataset_type:
        print(f"  - Data augmentation: {status} ({dataset_type})")
    else:
        print(f"  - Data augmentation: {status}")


def get_dataset_name_from_path(dataset_path):
    """
    Extract dataset name from dataset path
    Args:
        dataset_path: Dataset path, e.g. 'data/ijmond_data/train'
    Returns:
        str: Dataset name, e.g. 'ijmond_data_train'
    """
    # Remove trailing slash and normalize path
    path = os.path.normpath(dataset_path.rstrip("/"))
    path_parts = path.split(os.sep)

    # Remove common meaningless parts
    filtered_parts = []
    skip_words = ["data", "dataset", "datasets"]

    for part in path_parts:
        if part.lower() not in skip_words and part.strip():
            filtered_parts.append(part)

    # If nothing left after filtering, use last two parts of original path
    if not filtered_parts:
        filtered_parts = path_parts[-2:] if len(path_parts) >= 2 else path_parts[-1:]

    # Build dataset name
    dataset_name = "_".join(filtered_parts)

    # Clean name, keep only letters, numbers, underscore, hyphen
    dataset_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in dataset_name)

    return dataset_name


def get_loader(
    image_root,
    gt_root,
    trans_map_root,
    batchsize,
    trainsize,
    aug=True,
    freeze=False,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    random_seed=42,
    dataset_type="",
):
    """
    Create single training data loader (for weaklysupervised learning)
    Args:
        image_root: Image root directory
        gt_root: Ground truth root directory
        trans_map_root: Transmission map root directory
        batchsize: Batch size
        trainsize: Training image size
        aug: Whether to enable data augmentation, default True
        freeze: Whether to freeze all randomness for full reproducibility, default False
        shuffle: Whether to shuffle training data (overridden to False if freeze=True)
        num_workers: Number of data loading threads
        pin_memory: Whether to pin memory
        random_seed: Random seed
        dataset_type: Dataset type description, for enhanced logging
    Returns:
        DataLoader: Training data loader
    """
    # Unified randomness setting
    actual_shuffle = _setup_reproducibility(freeze, random_seed) and shuffle

    # Get file paths and create dataset
    images, gts, trans = _get_file_paths(image_root, gt_root, trans_map_root)
    dataset = SalObjDataset(images, gts, trans, trainsize, aug=aug, freeze=freeze)

    print(f"Dataset size: {len(dataset)}")
    _print_augmentation_status(aug, freeze, dataset_type)

    # Create data loader
    train_loader = data.DataLoader(
        dataset, batch_size=batchsize, shuffle=actual_shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )

    return train_loader


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
    num_workers=8,
    pin_memory=True,
    random_seed=42,
):
    """
    Create training and validation data loaders (compatible with fully-supervised module)
    Args:
        image_root: Image root directory
        gt_root: Ground truth root directory
        trans_map_root: Transmission map root directory
        batchsize: Batch size
        trainsize: Training image size
        val_split: Validation set ratio, default 0.2 (20%)
        aug: Whether to enable data augmentation for training set, default True
        freeze: Whether to freeze all randomness for full reproducibility, default False
        shuffle: Whether to shuffle training data (overridden to False if freeze=True)
        num_workers: Number of data loading threads
        pin_memory: Whether to pin memory
        random_seed: Random seed
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Unified randomness setting
    actual_shuffle = _setup_reproducibility(freeze, random_seed) and shuffle

    # Get file paths
    images, gts, trans = _get_file_paths(image_root, gt_root, trans_map_root)

    # Create index list and split into training and validation sets
    indices = list(range(len(images)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=random_seed, shuffle=True)

    # Create datasets
    train_dataset = SalObjDataset.from_indices(images, gts, trans, train_indices, trainsize, aug=aug, freeze=freeze)
    val_dataset = SalObjDataset.from_indices(images, gts, trans, val_indices, trainsize, aug=False, freeze=freeze)  # No augmentation for validation

    print(f"Total dataset size: {len(images)}")
    print(f"Training set size: {len(train_dataset)}")
    _print_augmentation_status(aug, freeze, "training data")
    print(f"Validation set size: {len(val_dataset)}")
    _print_augmentation_status(False, freeze, "validation data")  # Validation set never uses augmentation

    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=actual_shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    return train_loader, val_loader


class SalObjDataset(data.Dataset):
    """Unified saliency object dataset class"""

    def __init__(self, all_images, all_gts, all_trans, trainsize, indices=None, aug=False, freeze=False):
        """
        Args:
            all_images: List of all image paths
            all_gts: List of all GT paths
            all_trans: List of all transmission map paths
            trainsize: Training image size
            indices: Index list, if None use all data
            aug: Whether to enable data augmentation
            freeze: Whether to freeze randomness
        """
        self.trainsize = trainsize
        self.aug = aug
        self.freeze = freeze

        # Select data by index
        if indices is not None:
            self.images = [all_images[i] for i in indices]
            self.gts = [all_gts[i] for i in indices]
            self.trans = [all_trans[i] for i in indices]
        else:
            self.images = all_images
            self.gts = all_gts
            self.trans = all_trans

        # Filter unmatched files
        self._filter_files()
        self.size = len(self.images)

        # Initialize data transforms
        self._setup_transforms()

    @classmethod
    def from_indices(cls, all_images, all_gts, all_trans, indices, trainsize, aug=False, freeze=False):
        """
        Class method: create dataset instance from indices
        """
        return cls(all_images, all_gts, all_trans, trainsize, indices=indices, aug=aug, freeze=freeze)

    def _setup_transforms(self):
        """Set up data transforms"""
        # Color augmentation (only applied if needed)
        if self.aug and not self.freeze:
            self.img_color_transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                ]
            )
        else:
            self.img_color_transform = None
            if self.freeze and self.aug:
                print("[FREEZE MODE] Data augmentation disabled for full reproducibility")

        # Basic transforms (always applied)
        self.img_basic_transform = transforms.Compose(
            [
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # GT and trans transforms
        self.gt_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
        self.trans_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):
        """Get data item"""
        image = self._rgb_loader(self.images[index])
        gt = self._binary_loader(self.gts[index])
        trans = self._binary_loader(self.trans[index])

        # Apply color transform (image only)
        if self.img_color_transform is not None:
            image = self.img_color_transform(image)

        # Apply basic transforms
        image = self.img_basic_transform(image)
        gt = self.gt_transform(gt)
        trans = self.trans_transform(trans)

        return image, gt, trans

    def _filter_files(self):
        """Filter files with unmatched sizes"""
        assert len(self.images) == len(self.gts) == len(self.trans), "Number of images, GTs, and transmission maps do not match"

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
        """Load RGB image"""
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def _binary_loader(self, path):
        """Load binary image"""
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

    def grayscale_loader(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img
