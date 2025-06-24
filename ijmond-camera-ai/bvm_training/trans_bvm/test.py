import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from dataloader import test_dataset
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--langevin_step_num_des", type=int, default=10, help="number of langevin steps for ebm")
parser.add_argument("-langevin_step_size_des", type=float, default=0.026, help="step size of EBM langevin")
parser.add_argument("--energy_form", default="identity", help="tanh | sigmoid | identity | softplus")
parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")
parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
parser.add_argument("--model_path", type=str, required=True, help="path to model file", default="models/ucnet_trans3/Model_50_gen.pth")
parser.add_argument(
    "--method",
    type=str,
    required=True,
    default="supervised",
    choices=["supervised", "semi_supervised", "domain_adaptation"],
    help="training method: supervised | semi_supervised | domain_adaptation",
)
parser.add_argument("--test_dataset", type=str, required=True, choices=["ijmond", "smoke5k"], help="test dataset: ijmond | smoke5k")
opt = parser.parse_args()

# 根据测试数据集设置数据路径
if opt.test_dataset == "ijmond":
    dataset_path = "data/ijmond_data/test/img/"
elif opt.test_dataset == "smoke5k":
    dataset_path = "data/SMOKE5K_Dataset/SMOKE5K/test/img/"

# 检测设备并设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)

# 鲁棒的模型加载
print("Loading model...")
try:
    if device.type == "cuda":
        state_dict = torch.load(opt.model_path, weights_only=True)
    else:
        state_dict = torch.load(opt.model_path, map_location="cpu")

    # 过滤掉不匹配的键
    model_dict = generator.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and state_dict[k].shape == model_dict[k].shape}

    print(f"Successfully loaded {len(filtered_dict)}/{len(state_dict)} parameters")
    if len(filtered_dict) != len(state_dict):
        print("Warning: Some parameters were not loaded due to architecture mismatch")
        missing_keys = set(state_dict.keys()) - set(filtered_dict.keys())
        print(f"Missing keys: {list(missing_keys)[:5]}...")  # 只显示前5个

    generator.load_state_dict(filtered_dict, strict=False)

except Exception as e:
    print(f"Model loading failed: {e}")
    print("Please check if the model file and Generator architecture match")
    exit(1)

generator.to(device)
generator.eval()

# test_datasets = ['CAMO','CHAMELEON','COD10K']
test_datasets = [""]


def compute_energy(disc_score):
    if opt.energy_form == "tanh":
        energy = torch.tanh(-disc_score.squeeze())
    elif opt.energy_form == "sigmoid":
        energy = F.sigmoid(-disc_score.squeeze())
    elif opt.energy_form == "identity":
        energy = -disc_score.squeeze()
    elif opt.energy_form == "softplus":
        energy = F.softplus(-disc_score.squeeze())
    return energy


for dataset in test_datasets:
    # 从模型路径中提取完整的模型信息
    model_rel_path = opt.model_path.replace("models/", "") if opt.model_path.startswith("models/") else opt.model_path
    model_name = os.path.splitext(os.path.basename(model_rel_path))[0]  # 不带扩展名的文件名
    model_dir = os.path.dirname(model_rel_path)  # 父目录路径

    # 构建包含完整模型信息的保存路径，避免同目录下多个模型混淆
    if model_dir and model_dir != ".":
        # 包含目录和具体模型名: ./results/method/dataset/model_dir/model_name/
        save_path = os.path.join("./results", opt.method, opt.test_dataset, model_dir, model_name, dataset)
    else:
        # 只有模型名: ./results/method/dataset/model_name/
        save_path = os.path.join("./results", opt.method, opt.test_dataset, model_name, dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = os.path.join(dataset_path, dataset) if dataset else dataset_path
    print(f"Loading test data from: {image_root}")

    # 检查路径是否存在
    if not os.path.exists(image_root):
        print(f"Warning: Path {image_root} does not exist, skipping...")
        continue

    # 确定GT路径
    if opt.test_dataset == "ijmond":
        gt_root = os.path.join("data/ijmond_data/test/gt", dataset) if dataset else "data/ijmond_data/test/gt/"
    elif opt.test_dataset == "smoke5k":
        gt_root = os.path.join("data/SMOKE5K_Dataset/SMOKE5K/test/gt_", dataset) if dataset else "data/SMOKE5K_Dataset/SMOKE5K/test/gt_/"

    # 初始化评价指标统计变量
    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    sum_TN = 0
    total_images = 0

    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(f"Processing image {i+1}/{test_loader.size}")
        image, HH, WW, name = test_loader.load_data()
        image = image.to(device)  # 使用设备无关的方法
        generator_pred = generator.forward(image, training=False)
        res = generator_pred
        res = F.interpolate(res, size=[WW, HH], mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 安全的文件保存
        save_file_path = os.path.join(save_path, name)
        try:
            cv2.imwrite(save_file_path, res)
        except Exception as e:
            print(f"Failed to save {save_file_path}: {e}")

        # 计算评价指标（如果GT路径存在）
        if gt_root is not None:
            gt_path = os.path.join(gt_root, name)
            if os.path.exists(gt_path):
                # 读取GT图像
                gt_mask = cv2.imread(gt_path, 0)
                if gt_mask is not None:
                    # 二值化GT和预测结果
                    gt_bin = (gt_mask > 128).astype(np.uint8)
                    pred_bin = (res > 128).astype(np.uint8)

                    # 计算TP, FP, FN, TN
                    TP = np.logical_and(pred_bin, gt_bin).sum()
                    FP = np.logical_and(pred_bin, 1 - gt_bin).sum()
                    FN = np.logical_and(1 - pred_bin, gt_bin).sum()
                    TN = np.logical_and(1 - pred_bin, 1 - gt_bin).sum()

                    sum_TP += TP
                    sum_FP += FP
                    sum_FN += FN
                    sum_TN += TN
                    total_images += 1
            else:
                print(f"Warning: GT file {gt_path} not found")

    # 计算并保存评价指标
    if total_images > 0:
        # 计算各项指标
        precision = sum_TP / (sum_TP + sum_FP + 1e-8)
        recall = sum_TP / (sum_TP + sum_FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = sum_TN / (sum_TN + sum_FP + 1e-8)
        accuracy = (sum_TP + sum_TN) / (sum_TP + sum_TN + sum_FP + sum_FN + 1e-8)

        # IoU 计算
        iou_positive = sum_TP / (sum_TP + sum_FP + sum_FN + 1e-8)  # 烟雾类 IoU
        iou_negative = sum_TN / (sum_TN + sum_FP + sum_FN + 1e-8)  # 背景类 IoU
        miou = (iou_positive + iou_negative) / 2  # 真正的 mIoU：两个类别 IoU 的平均

        # Dice 系数（与 F1-Score 等价）
        dice = 2 * sum_TP / (2 * sum_TP + sum_FP + sum_FN + 1e-8)

        # 打印结果
        print(f"\n=== Evaluation Results for {opt.test_dataset} dataset ===")
        print(f"Processed {total_images} images")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"IoU (Smoke): {iou_positive:.4f}")
        print(f"IoU (Background): {iou_negative:.4f}")
        print(f"mIoU: {miou:.4f}")
        print(f"Dice Coefficient: {dice:.4f}")

        # 混淆矩阵
        confusion_matrix = f"""
Confusion Matrix:
                Predicted
                 P    N
Actual P    {sum_TP:8d} {sum_FN:8d}
Actual N    {sum_FP:8d} {sum_TN:8d}
"""
        print(confusion_matrix)

        # 保存评价指标到txt文件
        metrics_file = os.path.join(save_path, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Evaluation Results for {opt.test_dataset} dataset\n")
            f.write(f"Test Dataset: {opt.test_dataset}\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write(f"GT Path: {gt_root}\n")
            f.write(f"Model: {opt.model_path}\n")
            f.write(f"Method: {opt.method}\n")
            f.write(f"Test size: {opt.testsize}\n")
            f.write(f"Total images processed: {total_images}\n\n")

            f.write("Evaluation Metrics:\n")
            f.write(f"Precision: {precision:.6f}\n")
            f.write(f"Recall: {recall:.6f}\n")
            f.write(f"F1-Score: {f1_score:.6f}\n")
            f.write(f"Specificity: {specificity:.6f}\n")
            f.write(f"Accuracy: {accuracy:.6f}\n")
            f.write(f"IoU (Smoke): {iou_positive:.6f}\n")
            f.write(f"IoU (Background): {iou_negative:.6f}\n")
            f.write(f"mIoU: {miou:.6f}\n")
            f.write(f"Dice Coefficient: {dice:.6f}\n\n")

            f.write("Confusion Matrix:\n")
            f.write("                Predicted\n")
            f.write("                 P    N\n")
            f.write(f"Actual P    {sum_TP:8d} {sum_FN:8d}\n")
            f.write(f"Actual N    {sum_FP:8d} {sum_TN:8d}\n\n")

            f.write("Raw counts:\n")
            f.write(f"True Positives (TP): {sum_TP}\n")
            f.write(f"False Positives (FP): {sum_FP}\n")
            f.write(f"False Negatives (FN): {sum_FN}\n")
            f.write(f"True Negatives (TN): {sum_TN}\n")

        print(f"Evaluation metrics saved to: {metrics_file}")
    else:
        print("No valid GT images found for evaluation")
