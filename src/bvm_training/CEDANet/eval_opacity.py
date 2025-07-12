import os
import cv2
import numpy as np


def get_stem(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def get_matched_pairs(pred_dir, gt_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
    pred_stems = {get_stem(f): f for f in pred_files}
    gt_stems = {get_stem(f): f for f in gt_files}
    common = set(pred_stems.keys()) & set(gt_stems.keys())
    pairs = [(os.path.join(pred_dir, pred_stems[k]), os.path.join(gt_dir, gt_stems[k])) for k in sorted(common)]
    return pairs


def compute_iou(pred, gt):
    inter = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou = inter.sum() / (union.sum() + 1e-8)
    return iou


def main():
    pred_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../results/thesis/ijmond/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_data_train_da/SMOKE5K_Dataset_SMOKE5K_train_to_ijmond_data_train_da_best_model",
        )
    )
    gt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/ijmond_data/test/gt_3cls"))
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]

    # Write to txt file
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs_tri"))
    os.makedirs(out_dir, exist_ok=True)

    NAME = os.path.join(out_dir, "golden_tri.txt")

    print(f"Prediction folder: {pred_dir}")
    print(f"Label folder: {gt_dir}")
    print(f"Number of prediction images: {len(pred_files)}")
    print(f"Number of label images: {len(gt_files)}")
    pairs = get_matched_pairs(pred_dir, gt_dir)
    print(f"Found {len(pairs)} pairs of images with the same name")
    if len(pairs) == 0:
        print("No matching images found, please check file names and paths!")
        return
    # New statistics variables
    n_pred_fg_1 = 0  # Predicted as foreground and gt is class 1
    n_pred_fg_2 = 0  # Predicted as foreground and gt is class 2
    n_pred_fg_bg = 0  # Predicted as foreground and gt is background
    n_pred_fg_total = 0  # Total predicted foreground pixels
    n_gt_1 = 0  # Total gt class 1 pixels
    n_gt_2 = 0  # Total gt class 2 pixels
    n_gt_bg = 0  # Total gt background pixels
    n_gt_1_pred_fg = 0  # gt class 1 and predicted as foreground
    n_gt_2_pred_fg = 0  # gt class 2 and predicted as foreground
    n_gt_bg_pred_fg = 0  # gt background and predicted as foreground

    # IoU accumulators
    iou_1_sum = 0
    iou_2_sum = 0
    iou_bg_sum = 0
    n_img_1 = 0
    n_img_2 = 0
    n_img_bg = 0

    # MSE and F1 accumulators
    mse_1_sum = 0
    mse_2_sum = 0
    f1_1_sum = 0
    f1_2_sum = 0

    for pred_path, gt_path in pairs:
        print(f"Processing: pred={os.path.basename(pred_path)}, gt={os.path.basename(gt_path)}")
        pred = cv2.imread(pred_path, 0)
        gt = cv2.imread(gt_path, 0)
        if pred is None or gt is None:
            print(f"Skip: {pred_path}, {gt_path}")
            continue
        pred_bin = (pred >= 127).astype(np.uint8)
        # Three-class masks
        mask_pred_fg = pred_bin == 1
        mask_pred_bg = pred_bin == 0
        mask_gt_1 = gt == 255
        mask_gt_2 = (gt != 0) & (gt != 255)
        mask_gt_bg = gt == 0
        # Distribution of gt in predicted foreground pixels
        n_pred_fg_1 += np.logical_and(mask_pred_fg, mask_gt_1).sum()
        n_pred_fg_2 += np.logical_and(mask_pred_fg, mask_gt_2).sum()
        n_pred_fg_bg += np.logical_and(mask_pred_fg, mask_gt_bg).sum()
        n_pred_fg_total += mask_pred_fg.sum()
        # Total pixels for each gt class
        n_gt_1 += mask_gt_1.sum()
        n_gt_2 += mask_gt_2.sum()
        n_gt_bg += mask_gt_bg.sum()
        # Pixels in each gt class predicted as foreground
        n_gt_1_pred_fg += np.logical_and(mask_gt_1, mask_pred_fg).sum()
        n_gt_2_pred_fg += np.logical_and(mask_gt_2, mask_pred_fg).sum()
        n_gt_bg_pred_fg += np.logical_and(mask_gt_bg, mask_pred_fg).sum()

        # IoU for class 1
        if mask_gt_1.sum() > 0:
            inter_1 = np.logical_and(mask_pred_fg, mask_gt_1).sum()
            union_1 = np.logical_or(mask_pred_fg, mask_gt_1).sum()
            iou_1_sum += inter_1 / (union_1 + 1e-8)

            # MSE for class 1 (high concentration smoke area: binary prediction vs GT=255)
            pred_1_area = pred[mask_gt_1].astype(np.float32) / 255.0  # Prediction in high concentration area: 0 or 1
            gt_1_target = np.ones_like(pred_1_area)  # Target in high concentration area should be 1 (smoke)
            mse_1_sum += np.mean((pred_1_area - gt_1_target) ** 2)

            # F1 for class 1
            tp_1 = inter_1
            fp_1 = mask_pred_fg.sum() - tp_1
            fn_1 = mask_gt_1.sum() - tp_1
            precision_1 = tp_1 / (tp_1 + fp_1 + 1e-8)
            recall_1 = tp_1 / (tp_1 + fn_1 + 1e-8)
            f1_1_sum += 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-8)

            n_img_1 += 1
        # IoU for class 2
        if mask_gt_2.sum() > 0:
            inter_2 = np.logical_and(mask_pred_fg, mask_gt_2).sum()
            union_2 = np.logical_or(mask_pred_fg, mask_gt_2).sum()
            iou_2_sum += inter_2 / (union_2 + 1e-8)

            # MSE for class 2 (low concentration smoke area: binary prediction vs ideal target)
            pred_2_area = pred[mask_gt_2].astype(np.float32) / 255.0  # Prediction in low concentration area: 0 or 1
            gt_2_target = np.ones_like(pred_2_area)  # Low concentration should also be predicted as 1 (smoke)
            mse_2_sum += np.mean((pred_2_area - gt_2_target) ** 2)

            # F1 for class 2
            tp_2 = inter_2
            fp_2 = mask_pred_fg.sum() - tp_2
            fn_2 = mask_gt_2.sum() - tp_2
            precision_2 = tp_2 / (tp_2 + fp_2 + 1e-8)
            recall_2 = tp_2 / (tp_2 + fn_2 + 1e-8)
            f1_2_sum += 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-8)

            n_img_2 += 1
        # IoU for background
        if mask_gt_bg.sum() > 0:
            inter_bg = np.logical_and(mask_pred_bg, mask_gt_bg).sum()
            union_bg = np.logical_or(mask_pred_bg, mask_gt_bg).sum()
            iou_bg_sum += inter_bg / (union_bg + 1e-8)
            n_img_bg += 1

    print(f"\n=== Ternary Label Foreground Capture Statistics ===")
    print(f"Total predicted foreground pixels: {n_pred_fg_total}")
    print(f"  Foreground class-1 (gt==255): {n_pred_fg_1} ({n_pred_fg_1/n_pred_fg_total:.2%} of pred_fg)")
    print(f"  Foreground class-2 (0<gt<255): {n_pred_fg_2} ({n_pred_fg_2/n_pred_fg_total:.2%} of pred_fg)")
    print(f"  Background (gt==0): {n_pred_fg_bg} ({n_pred_fg_bg/n_pred_fg_total:.2%} of pred_fg)")
    print(f"\n=== Per-class Recall (predicted as foreground) ===")
    print(f"Class-1 (gt==255) Recall: {n_gt_1_pred_fg}/{n_gt_1} = {n_gt_1_pred_fg/n_gt_1:.2%}")
    print(f"Class-2 (0<gt<255) Recall: {n_gt_2_pred_fg}/{n_gt_2} = {n_gt_2_pred_fg/n_gt_2:.2%}")
    print(f"Background (gt==0) False Positive Rate: {n_gt_bg_pred_fg}/{n_gt_bg} = {n_gt_bg_pred_fg/n_gt_bg:.2%}")

    print(f"\n=== Per-class IoU (image-wise mean) ===")
    mean_iou_1 = iou_1_sum / n_img_1 if n_img_1 > 0 else 0
    mean_iou_2 = iou_2_sum / n_img_2 if n_img_2 > 0 else 0
    mean_iou_bg = iou_bg_sum / n_img_bg if n_img_bg > 0 else 0
    miou_3cls = (mean_iou_1 + mean_iou_2 + mean_iou_bg) / 3
    miou_fg = (mean_iou_1 + mean_iou_2) / 2
    print(f"Class-1 IoU: {mean_iou_1:.4f} (gt==255)")
    print(f"Class-2 IoU: {mean_iou_2:.4f} (0<gt<255)")
    print(f"Background IoU: {mean_iou_bg:.4f} (gt==0)")
    print(f"Ternary mIoU: {miou_3cls:.4f}")
    print(f"Foreground mIoU: {miou_fg:.4f} (mean of class-1 and class-2)")

    print(f"\n=== Per-class MSE (image-wise mean) ===")
    mean_mse_1 = mse_1_sum / n_img_1 if n_img_1 > 0 else 0
    mean_mse_2 = mse_2_sum / n_img_2 if n_img_2 > 0 else 0
    mmse_fg = (mean_mse_1 + mean_mse_2) / 2
    print(f"Class-1 MSE: {mean_mse_1:.6f} (gt==255)")
    print(f"Class-2 MSE: {mean_mse_2:.6f} (0<gt<255)")
    print(f"Foreground mMSE: {mmse_fg:.6f} (mean of class-1 and class-2)")

    print(f"\n=== Per-class F1 Score (image-wise mean) ===")
    mean_f1_1 = f1_1_sum / n_img_1 if n_img_1 > 0 else 0
    mean_f1_2 = f1_2_sum / n_img_2 if n_img_2 > 0 else 0
    mf1_fg = (mean_f1_1 + mean_f1_2) / 2
    print(f"Class-1 F1: {mean_f1_1:.4f} (gt==255)")
    print(f"Class-2 F1: {mean_f1_2:.4f} (0<gt<255)")
    print(f"Foreground mF1: {mf1_fg:.4f} (mean of class-1 and class-2)")

    with open(NAME, "w") as f:
        f.write("=== Ternary Label Foreground Capture Statistics ===\n")
        f.write(f"Total predicted foreground pixels: {n_pred_fg_total}\n")
        f.write(f"  Foreground class-1 (gt==255): {n_pred_fg_1} ({n_pred_fg_1/n_pred_fg_total:.2%} of pred_fg)\n")
        f.write(f"  Foreground class-2 (0<gt<255): {n_pred_fg_2} ({n_pred_fg_2/n_pred_fg_total:.2%} of pred_fg)\n")
        f.write(f"  Background (gt==0): {n_pred_fg_bg} ({n_pred_fg_bg/n_pred_fg_total:.2%} of pred_fg)\n")
        f.write("\n=== Per-class Recall (predicted as foreground) ===\n")
        f.write(f"Class-1 (gt==255) Recall: {n_gt_1_pred_fg}/{n_gt_1} = {n_gt_1_pred_fg/n_gt_1:.2%}\n")
        f.write(f"Class-2 (0<gt<255) Recall: {n_gt_2_pred_fg}/{n_gt_2} = {n_gt_2_pred_fg/n_gt_2:.2%}\n")
        f.write(f"Background (gt==0) False Positive Rate: {n_gt_bg_pred_fg}/{n_gt_bg} = {n_gt_bg_pred_fg/n_gt_bg:.2%}\n")
        f.write("\n=== Per-class IoU (image-wise mean) ===\n")
        f.write(f"Class-1 IoU: {mean_iou_1:.4f} (gt==255)\n")
        f.write(f"Class-2 IoU: {mean_iou_2:.4f} (0<gt<255)\n")
        f.write(f"Background IoU: {mean_iou_bg:.4f} (gt==0)\n")
        f.write(f"Ternary mIoU: {miou_3cls:.4f}\n")
        f.write(f"Foreground mIoU: {miou_fg:.4f} (mean of class-1 and class-2)\n")
        f.write("\n=== Per-class MSE (image-wise mean) ===\n")
        f.write(f"Class-1 MSE: {mean_mse_1:.6f} (gt==255)\n")
        f.write(f"Class-2 MSE: {mean_mse_2:.6f} (0<gt<255)\n")
        f.write(f"Foreground mMSE: {mmse_fg:.6f} (mean of class-1 and class-2)\n")
        f.write("\n=== Per-class F1 Score (image-wise mean) ===\n")
        f.write(f"Class-1 F1: {mean_f1_1:.4f} (gt==255)\n")
        f.write(f"Class-2 F1: {mean_f1_2:.4f} (0<gt<255)\n")
        f.write(f"Foreground mF1: {mf1_fg:.4f} (mean of class-1 and class-2)\n")
    print(f"\nResults saved to: {NAME}")


if __name__ == "__main__":
    main()
