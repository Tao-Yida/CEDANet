# from PIL import Image
# import numpy as np
# import time
# import torch
# from torch import nn, autograd, optim, Tensor, cuda
# from torch.nn import functional as F
# from copy import deepcopy

# def normalize_(array):
#     """Normalize the array to the range [0, 1]."""
#     min_val = np.min(array)
#     max_val = np.max(array)
#     return (array - min_val) / (max_val - min_val)

# def mse(preds, labels, th=0.5, mask=None):
#     preds_normalized = normalize_(preds)
#     labels_normalized = normalize_(labels)

#     if mask is not None:
#         # print(np.sum(preds>0))
#         preds = preds[mask]
#         # print(np.sum(preds>0))
#         labels = labels[mask]

#     return np.mean((preds_normalized - labels_normalized) ** 2)

# def fscore(preds, labels, th=0.5, mask=None):
#     """
#     Calculate the F-score for the predicted and ground truth masks.
    
#     Parameters:
#     - preds: np.ndarray, the predicted mask.
#     - labels: np.ndarray, the ground truth mask.
#     - th: float, threshold for converting prediction probabilities to binary (default is 0.5).
#     - mask: np.ndarray or None, optional mask specifying the area of interest.
#             If provided, only the masked area will be considered for calculation.
            
#     Returns:
#     - F: float, the calculated F-score.
#     """
#     # Normalize predictions and labels
#     preds = normalize_(preds)
#     labels = normalize_(labels)

#     # Apply mask if provided
#     if mask is not None:
#         # print(np.sum(preds>0))
#         preds = preds[mask]
#         # print(np.sum(preds>0))
#         labels = labels[mask]

#     # Convert labels to boolean
#     labels_bool = labels.astype(bool)

#     # Threshold predictions to create binary predictions
#     tmp = preds >= th

#     # Calculate True Positives, total predicted positives, and total true positives
#     TP = np.sum(tmp & labels_bool)
#     T1 = np.sum(tmp)
#     T2 = np.sum(labels_bool)

#     # Calculate the F-score with a specific formula
#     F = 1.3 * TP / (T1 + 0.3 * T2 + 1e-9)

#     return F

# def calculate_iou(preds, labels, th=0.5, mask=None):
#     # Ensure the masks are boolean arrays
#     preds = normalize_(preds)
#     labels = normalize_(labels)

#     if mask is not None:
#         # print(np.sum(preds>0))
#         preds = preds[mask]
#         # print(np.sum(preds>0))
#         labels = labels[mask]

#     mask1 = preds >= th
#     mask2 = labels.astype(bool)
    
#     # Calculate intersection and union
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
    
#     # Calculate IoU
#     iou = intersection / union if union != 0 else 0
#     return iou


from PIL import Image
import numpy as np
import time
import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from copy import deepcopy

def normalize_(array):
    """Normalize the array to the range [0, 1]."""
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

def mse(preds, labels, th=0.5, mask_labels=None, mask_preds=None):
    preds_normalized = normalize_(preds)
    labels_normalized = normalize_(labels)

    if mask_labels is not None:
        labels[mask_labels] = 1
        labels[~mask_labels] = 0
    if mask_preds is not None:
        preds[mask_preds] = 0 # hide the ground truth region of the opacity that you dont want to calculate from the predictions
        # in that way you hide tha prediction that the model did 

    return np.mean((preds_normalized - labels_normalized) ** 2)

def fscore(preds, labels, th=0.5, mask_labels=None, mask_preds=None):
    """
    Calculate the F-score for the predicted and ground truth masks.
    
    Parameters:
    - preds: np.ndarray, the predicted mask.
    - labels: np.ndarray, the ground truth mask.
    - th: float, threshold for converting prediction probabilities to binary (default is 0.5).
    - mask: np.ndarray or None, optional mask specifying the area of interest.
            If provided, only the masked area will be considered for calculation.
            
    Returns:
    - F: float, the calculated F-score.
    """
    # Normalize predictions and labels
    preds = normalize_(preds)
    labels = normalize_(labels)

    # Apply mask if provided
    if mask_labels is not None:
        labels[mask_labels] = 1
        labels[~mask_labels] = 0
    if mask_preds is not None:
        preds[mask_preds] = 0 # hide the ground truth region of the opacity that you dont want to calculate from the predictions
        # in that way you hide tha prediction that the model did 

    # Convert labels to boolean
    labels_bool = labels.astype(bool)

    # Threshold predictions to create binary predictions
    tmp = preds >= th

    # Calculate True Positives, total predicted positives, and total true positives
    TP = np.sum(tmp & labels_bool)
    T1 = np.sum(tmp)
    T2 = np.sum(labels_bool)

    # Calculate the F-score with a specific formula
    F = 1.3 * TP / (T1 + 0.3 * T2 + 1e-9)

    return F

def calculate_iou(preds, labels, th=0.5, mask_labels=None, mask_preds=None):
    # Ensure the masks are boolean arrays
    preds = normalize_(preds)
    labels = normalize_(labels)

    # Apply mask if provided
    if mask_labels is not None:
        labels[mask_labels] = 1
        labels[~mask_labels] = 0
    if mask_preds is not None:
        preds[mask_preds] = 0 # hide the ground truth region of the opacity that you dont want to calculate from the predictions
        # in that way you hide tha prediction that the model did 

    mask1 = preds >= th
    mask2 = labels.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou