import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")
parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
parser.add_argument("--model_path", type=str, default="./models/domain_adapt_smoke5k_ijmond/domain_adapt_epoch_45.pth", help="path to trained model")
parser.add_argument("--test_datasets", type=str, nargs="+", default=["ijmond", "smoke5k"], help="test datasets")

opt = parser.parse_args()

# 数据集路径配置 - 模仿半监督测试脚本
dataset_paths = {
    "ijmond": "data/ijmond_data/test/img/",
    "smoke5k": "data/SMOKE5K_Dataset/SMOKE5K/test/img/",
    "": "data/SMOKE5K_Dataset/SMOKE5K/test/img/",  # 默认
}

# 加载模型 - 完全模仿半监督测试脚本
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=16)

# 简单加载域适应模型权重
if os.path.exists(opt.model_path):
    checkpoint = torch.load(opt.model_path)

    # 如果是域适应模型，提取生成器权重
    if "model_state_dict" in checkpoint:
        generator_weights = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.startswith("base_generator."):
                new_key = key.replace("base_generator.", "")
                generator_weights[new_key] = value
        generator.load_state_dict(generator_weights)
    else:
        generator.load_state_dict(checkpoint)

    print(f"Model loaded from: {opt.model_path}")
else:
    print(f"Model not found: {opt.model_path}")
    exit(1)

generator.cuda()
generator.eval()

# 初始化累积指标
sum_TP = 0
sum_FP = 0
sum_FN = 0
total_images = 0


def compute_energy(disc_score):
    energy = -disc_score.squeeze()
    return energy


# 遍历测试数据集 - 完全模仿半监督测试脚本的循环结构
for dataset in opt.test_datasets:
    if dataset not in dataset_paths:
        print(f"Unknown dataset: {dataset}")
        continue

    # 设置保存路径 - 模仿半监督脚本的路径结构
    save_path = "./results/domain_adapt/" + dataset + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 设置数据路径
    dataset_path = dataset_paths[dataset]
    image_root = dataset_path + dataset if dataset else dataset_path

    # GT路径设置 - 模仿半监督脚本
    if "ijmond" in dataset_path:
        gt_root = dataset_path.replace("img/", "gt/") + dataset + "/"
    else:
        gt_root = dataset_path.replace("img/", "gt_/") + dataset + "/"

    print(f"Testing {dataset}...")
    print(f"Images: {image_root}")
    print(f"GT: {gt_root}")
    print(f"Save: {save_path}")

    test_loader = test_dataset(image_root, opt.testsize)

    for i in range(test_loader.size):
        print(f"Processing image {i + 1}/{test_loader.size} in dataset {dataset}...")

        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()

        # 推理 - 完全模仿半监督测试脚本
        generator_pred = generator.forward(image, training=False)

        # 后处理 - 完全模仿半监督测试脚本
        res = generator_pred
        res = F.upsample(res, size=[WW, HH], mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 保存预测图片 - 完全模仿半监督测试脚本
        cv2.imwrite(save_path + name, res)

        # 计算指标 - 完全模仿半监督测试脚本
        gt_path = os.path.join(gt_root, name)
        gt_mask = cv2.imread(gt_path, 0)
        if gt_mask is None:
            continue

        gt_bin = (gt_mask > 128).astype(np.uint8)
        pred_bin = (res > 128).astype(np.uint8)

        # 计算TP, FP, FN - 完全模仿半监督测试脚本
        TP = np.logical_and(pred_bin, gt_bin).sum()
        FP = np.logical_and(pred_bin, 1 - gt_bin).sum()
        FN = np.logical_and(1 - pred_bin, gt_bin).sum()

        sum_TP += TP
        sum_FP += FP
        sum_FN += FN
        total_images += 1

# 计算并输出指标 - 完全模仿半监督测试脚本
precision = sum_TP / (sum_TP + sum_FP + 1e-8)
recall = sum_TP / (sum_TP + sum_FN + 1e-8)
f1_score = 2 * precision * recall / (precision + recall + 1e-8)
miou = sum_TP / (sum_TP + sum_FP + sum_FN + 1e-8)

print(f"Processed {total_images} images. Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, mIoU: {miou:.4f}")

# 保存结果到文件
model_name = os.path.basename(opt.model_path).replace(".pth", "")
results_file = f"./results/domain_adapt/test_results_{model_name}.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

with open(results_file, "w") as f:
    f.write(f"Domain Adapt Model Test Results\n")
    f.write(f"Model: {opt.model_path}\n")
    f.write(f"Test datasets: {opt.test_datasets}\n")
    f.write(f"Total images: {total_images}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1_score:.4f}\n")
    f.write(f"mIoU: {miou:.4f}\n")

print(f"Results saved to: {results_file}")
