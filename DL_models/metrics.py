import torch
import numpy as np


def calculate_metrics(outputs, masks, num_classes):
    """计算像素精度和 Mean IoU"""
    preds = torch.argmax(outputs, dim=1)  # Get predicted class for each pixel

    # Ensure masks and preds are on CPU for numpy operations if needed, and are integer types
    preds_flat = preds.cpu().flatten()
    masks_flat = masks.cpu().flatten()

    # Pixel Accuracy
    correct = (preds_flat == masks_flat).sum().item()
    total = masks_flat.numel()  # Total number of pixels
    pixel_accuracy = correct / total if total > 0 else 0

    # Mean IoU
    iou_list = []
    present_classes_iou = (
        []
    )  # Store IoU only for classes present in the ground truth mask
    for cls in range(num_classes):
        pred_inds = preds_flat == cls  # Use flattened tensors
        target_inds = masks_flat == cls  # Use flattened tensors
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            # If neither prediction nor target has this class, IoU is 1 (or NaN, often excluded)
            # If only one has it, IoU is 0. Let's treat union=0 as IoU=1 if intersection is also 0.
            iou = 1.0 if intersection == 0 else 0.0  # Or handle as NaN / exclude
        else:
            iou = intersection / union
        iou_list.append(iou)
        # Add to present list only if the class exists in the ground truth for this batch
        if target_inds.sum().item() > 0:
            present_classes_iou.append(iou)

    # mean_iou = np.mean(iou_list)  # IoU over all classes
    # More robust: Mean IoU over classes present in the ground truth of the batch
    mean_iou_present = np.mean(present_classes_iou) if present_classes_iou else 0.0

    # Return accuracy and the more robust mean IoU (over present classes)
    return pixel_accuracy, mean_iou_present
