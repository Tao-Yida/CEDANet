import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from dataloader import test_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cv2


def compute_mse(pred, gt):
    """Compute Mean Squared Error (MSE)"""
    pred_norm = pred.astype(np.float32) / 255.0
    gt_norm = gt.astype(np.float32) / 255.0
    return mean_squared_error(gt_norm.flatten(), pred_norm.flatten())


def compute_mae(pred, gt):
    """Compute Mean Absolute Error (MAE)"""
    pred_norm = pred.astype(np.float32) / 255.0
    gt_norm = gt.astype(np.float32) / 255.0
    return mean_absolute_error(gt_norm.flatten(), pred_norm.flatten())


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--latent_dim", type=int, default=8, help="latent dim")
parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
parser.add_argument("--model_path", type=str, required=True, help="path to model file", default="models/ucnet_trans3/Model_50_gen.pth")
parser.add_argument("--test_dataset", type=str, required=True, choices=["ijmond", "smoke5k"], help="test dataset: ijmond | smoke5k")
opt = parser.parse_args()

# Set data paths based on test dataset
if opt.test_dataset == "ijmond":
    dataset_path = "data/ijmond_data/test/img/"
elif opt.test_dataset == "smoke5k":
    dataset_path = "data/SMOKE5K_Dataset/SMOKE5K/test/img/"

# Detect device and set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)

# Robust model loading
print("Loading model...")
try:
    if device.type == "cuda":
        state_dict = torch.load(opt.model_path, weights_only=True)
    else:
        state_dict = torch.load(opt.model_path, map_location="cpu")

    # Filter out mismatched keys
    model_dict = generator.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and state_dict[k].shape == model_dict[k].shape}

    print(f"Successfully loaded {len(filtered_dict)}/{len(state_dict)} parameters")
    if len(filtered_dict) != len(state_dict):
        print("Warning: Some parameters were not loaded due to architecture mismatch")
        missing_keys = set(state_dict.keys()) - set(filtered_dict.keys())
        print(f"Missing keys: {list(missing_keys)[:5]}...")  # Only show first 5

    generator.load_state_dict(filtered_dict, strict=False)

except Exception as e:
    print(f"Model loading failed: {e}")
    print("Please check if the model file and Generator architecture match")
    exit(1)

generator.to(device)
generator.eval()

test_datasets = [""]


for dataset in test_datasets:
    # Extract complete model information from model path
    model_rel_path = opt.model_path.replace("models/", "") if opt.model_path.startswith("models/") else opt.model_path
    model_name = os.path.splitext(os.path.basename(model_rel_path))[0]  # Filename without extension
    model_dir = os.path.dirname(model_rel_path)  # Parent directory path

    # Build save path with complete model information to avoid confusion between multiple models in the same directory
    if model_dir and model_dir != ".":
        # Include directory and specific model name: ./results/supervised/dataset/model_dir/model_name/
        save_path = os.path.join("./results", "supervised", opt.test_dataset, model_dir, model_name, dataset)
    else:
        # Only model name: ./results/supervised/dataset/model_name/
        save_path = os.path.join("./results", "supervised", opt.test_dataset, model_name, dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = os.path.join(dataset_path, dataset) if dataset else dataset_path
    print(f"Loading test data from: {image_root}")

    # Check if path exists
    if not os.path.exists(image_root):
        print(f"Warning: Path {image_root} does not exist, skipping...")
        continue

    # Determine GT path
    if opt.test_dataset == "ijmond":
        gt_root = os.path.join("data/ijmond_data/test/gt", dataset) if dataset else "data/ijmond_data/test/gt/"
    elif opt.test_dataset == "smoke5k":
        gt_root = os.path.join("data/SMOKE5K_Dataset/SMOKE5K/test/gt_", dataset) if dataset else "data/SMOKE5K_Dataset/SMOKE5K/test/gt_/"

    # Initialize evaluation metric statistical variables
    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    sum_TN = 0
    total_images = 0

    # Initialize additional evaluation metric accumulation variables
    sum_mse = 0.0
    sum_mae = 0.0

    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(f"Processing image {i+1}/{test_loader.size}")
        image, HH, WW, name = test_loader.load_data()
        image = image.to(device)  # Use device-agnostic method
        generator_pred = generator.forward(image, training=False)
        res = generator_pred
        res = F.interpolate(res, size=[WW, HH], mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)

        # Safe file saving
        save_file_path = os.path.join(save_path, name)
        try:
            cv2.imwrite(save_file_path, res)
        except Exception as e:
            print(f"Failed to save {save_file_path}: {e}")

        # Calculate evaluation metrics (if GT path exists)
        if gt_root is not None:
            gt_path = os.path.join(gt_root, name)
            if os.path.exists(gt_path):
                # Read GT image
                gt_mask = cv2.imread(gt_path, 0)
                if gt_mask is not None:
                    # Binarize GT and prediction results
                    gt_bin = (gt_mask > 128).astype(np.uint8)
                    pred_bin = (res > 128).astype(np.uint8)

                    # Calculate TP, FP, FN, TN
                    TP = np.logical_and(pred_bin, gt_bin).sum()
                    FP = np.logical_and(pred_bin, 1 - gt_bin).sum()
                    FN = np.logical_and(1 - pred_bin, gt_bin).sum()
                    TN = np.logical_and(1 - pred_bin, 1 - gt_bin).sum()

                    sum_TP += TP
                    sum_FP += FP
                    sum_FN += FN
                    sum_TN += TN
                    total_images += 1

                    # Calculate additional evaluation metrics
                    mse = compute_mse(res, gt_mask)
                    mae = compute_mae(res, gt_mask)

                    sum_mse += mse
                    sum_mae += mae
            else:
                print(f"Warning: GT file {gt_path} not found")

    # Calculate and save evaluation metrics
    if total_images > 0:
        # Calculate various metrics
        precision = sum_TP / (sum_TP + sum_FP + 1e-8)
        recall = sum_TP / (sum_TP + sum_FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = sum_TN / (sum_TN + sum_FP + 1e-8)
        accuracy = (sum_TP + sum_TN) / (sum_TP + sum_TN + sum_FP + sum_FN + 1e-8)

        # IoU calculation
        iou_positive = sum_TP / (sum_TP + sum_FP + sum_FN + 1e-8)  # Smoke class IoU
        iou_negative = sum_TN / (sum_TN + sum_FN + sum_FP + 1e-8)  # Background class IoU
        miou = (iou_positive + iou_negative) / 2  # True mIoU: average of two class IoUs

        # Dice coefficient (equivalent to F1-Score)
        dice = 2 * sum_TP / (2 * sum_TP + sum_FP + sum_FN + 1e-8)

        # Calculate average advanced metrics
        avg_mse = sum_mse / total_images
        avg_mae = sum_mae / total_images

        # Print results
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
        print(f"\n=== Additional Evaluation Metrics ===")
        print(f"Mean MSE: {avg_mse:.6f}")
        print(f"Mean MAE: {avg_mae:.6f}")

        # Confusion matrix
        confusion_matrix = f"""
Confusion Matrix:
                Predicted
                 P    N
Actual P    {sum_TP:8d} {sum_FN:8d}
Actual N    {sum_FP:8d} {sum_TN:8d}
"""
        print(confusion_matrix)

        # Save evaluation metrics to txt file
        metrics_file = os.path.join(save_path, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Evaluation Results for {opt.test_dataset} dataset\n")
            f.write(f"Test Dataset: {opt.test_dataset}\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write(f"GT Path: {gt_root}\n")
            f.write(f"Model: {opt.model_path}\n")
            f.write(f"Method: supervised\n")
            f.write(f"Test size: {opt.testsize}\n")
            f.write(f"Total images processed: {total_images}\n\n")

            f.write("Basic Evaluation Metrics:\n")
            f.write(f"Precision: {precision:.6f}\n")
            f.write(f"Recall: {recall:.6f}\n")
            f.write(f"F1-Score: {f1_score:.6f}\n")
            f.write(f"Specificity: {specificity:.6f}\n")
            f.write(f"Accuracy: {accuracy:.6f}\n")
            f.write(f"IoU (Smoke): {iou_positive:.6f}\n")
            f.write(f"IoU (Background): {iou_negative:.6f}\n")
            f.write(f"mIoU: {miou:.6f}\n")
            f.write(f"Dice Coefficient: {dice:.6f}\n\n")

            f.write("Additional Evaluation Metrics:\n")
            f.write(f"Mean MSE: {avg_mse:.8f}\n")
            f.write(f"Mean MAE: {avg_mae:.8f}\n\n")

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
