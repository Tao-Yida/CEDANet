"""
Video Pseudo Label Generator
===========================

This script is used to generate pseudo labels from video files for semi-supervised learning:
1. Load pretrained model
2. Extract frames from video
3. Predict on extracted frames
4. Select the frame with the highest confidence in each video and its n context frames before and after (2n+1 frames in total)
5. Output original images and corresponding pseudo labels
6. Generate corresponding transmission maps

Usage Example:
-------------
```
python inference.py --videos_path ../../data/videos \
                    --output_path ../../data/ijmond_camera \
                    --pretrained_weights ../../models/model.pth \
                    --sampling_rate 1 \
                    --context_frames 2 \
                    --threshold 0.5
```

Output Structure:
----------------
```
output_path/
    ├── img/    # Store original images (named: videoName_index.jpg)
    ├── gt/     # Store pseudo labels (named: videoName_index.png)
    └── trans/  # Store transmission maps (named: videoName_index.png)
```
"""

import torch
import torch.nn.functional as F
import os
import argparse
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
from pathlib import Path
import glob
import pandas as pd

# Add parent directory to system path to import modules from parent directory
sys.path.append(str(Path(__file__).parent.parent))
from trans_bvm.model.ResNet_models import Generator
from transmission_map import find_transmission_map
from smoothness import gradient_x, gradient_y, laplacian_edge


def arg_parse():
    """
    Parse command line arguments
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate pseudo labels from video for semi-supervised learning")
    parser.add_argument("--testsize", type=int, default=352, help="Test image size")
    parser.add_argument("--latent_dim", type=int, default=3, help="Latent space dimension")
    parser.add_argument("--feat_channel", type=int, default=32, help="Feature channel number")
    parser.add_argument("--videos_path", type=str, default="../../../data/ijmond_camera/videos", help="Video file path")
    parser.add_argument("--output_path", type=str, default="../../../data/ijmond_camera/pseudo_labels", help="Pseudo label output path")
    parser.add_argument(
        "--pretrained_weights", type=str, default="../../../models/finetune/Model_50_gen_no_pretrained_weights.pth", help="Pretrained weights path"
    )
    parser.add_argument("--sampling_rate", type=int, default=1, help="Frame sampling rate")
    parser.add_argument("--context_frames", type=int, default=2, help="Number of context frames before and after high-confidence frame")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pseudo label confidence threshold")
    parser.add_argument(
        "--constraint_type",
        type=str,
        choices=["none", "citizen", "expert"],
        default="none",
        help="Constraint type: none (no constraint), citizen (citizen constraint), expert (expert constraint)",
    )
    parser.add_argument("--video_labels_csv", type=str, default="../../../data/ijmond_camera/video_labels.csv", help="Video label CSV file path")
    opt = parser.parse_args()
    return opt


def load_video_labels(csv_path):
    """
    Load video label CSV file
    Args:
        csv_path: CSV file path
    Returns:
        dict: Video label dictionary, key is full file name (with extension), value is label info
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Video label file does not exist: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        video_labels = {}

        for _, row in df.iterrows():
            file_name = row["file_name"]
            label_state = row["label_state"]
            label_state_admin = row["label_state_admin"]

            # Use the complete file name as the key directly, without removing the suffix
            video_labels[file_name] = {"label_state": label_state, "label_state_admin": label_state_admin}

        print(f"Loaded {len(video_labels)} video segment labels")
        return video_labels

    except Exception as e:
        print(f"Error loading video labels: {e}")
        return {}


def get_video_constraint_info(video_name, video_labels, constraint_type):
    """
    Get video constraint information
    Args:
        video_name: Video name (complete file name, including suffix)
        video_labels: Video label dictionary
        constraint_type: Constraint type ('none', 'citizen', 'expert')
    Returns:
        dict: Dictionary containing constraint information
    """
    # If constraint type is none, return no constraint
    if constraint_type == "none":
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    # If video is not in label dictionary, return no constraint
    if video_name not in video_labels:
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    # Select label column according to constraint type
    if constraint_type == "citizen":
        label_key = "label_state"
    elif constraint_type == "expert":
        label_key = "label_state_admin"
    else:
        # Invalid constraint type
        print(f"Warning: Invalid constraint type '{constraint_type}', supported: 'none', 'citizen', 'expert'")
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    label_value = video_labels[video_name][label_key]

    # If label value is -1 (no data) or -2 (bad video), return no constraint
    if label_value == -1 or label_value == -2:
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    # Define label meanings and constraint strength (confidence corresponds to actual inference probability)
    label_meanings = {
        47: {"smoke": True, "confidence": 0.9, "strength": "gold_positive"},  # Gold standard positive sample -> inference probability 0.9
        32: {"smoke": False, "confidence": 0.2, "strength": "gold_negative"},  # Gold standard negative sample -> inference probability 0.2
        23: {"smoke": True, "confidence": 0.8, "strength": "strong_positive"},  # Strong positive sample -> inference probability 0.8
        16: {"smoke": False, "confidence": 0.4, "strength": "strong_negative"},  # Strong negative sample -> inference probability 0.4
        19: {"smoke": True, "confidence": 0.7, "strength": "weak_positive"},  # Weak positive sample -> inference probability 0.7
        20: {"smoke": False, "confidence": 0.5, "strength": "weak_negative"},  # Weak negative sample -> inference probability 0.5
        5: {"smoke": True, "confidence": 0.65, "strength": "maybe_positive"},  # Maybe positive sample -> inference probability 0.65
        4: {"smoke": False, "confidence": 0.58, "strength": "maybe_negative"},  # Maybe negative sample -> inference probability 0.58
        3: {"smoke": None, "confidence": 0.0, "strength": "discord"},  # Disagreement -> no constraint
    }

    if label_value in label_meanings:
        info = label_meanings[label_value]
        return {
            "has_constraint": True,
            "constraint_confidence": info["confidence"],
            "expected_smoke": info["smoke"],
            "constraint_strength": info["strength"],
        }
    else:
        # Unknown label value
        print(f"Warning: Unknown label value {label_value} for video {video_name}")
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}


def extract_frames_from_video(video_path, output_folder, sampling_rate=1):
    """
    Extract frames from video
    Args:
        video_path: Video file path
        output_folder: Output folder
        sampling_rate: Sampling rate, extract one frame every n frames
    Returns:
        dict: Frame info dictionary, contains frame path and frame index
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return {}

    frame_infos = {}
    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {video_path} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames according to sampling rate
        if frame_count % sampling_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:02d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_infos[frame_filename] = frame_count
            saved_count += 1

        frame_count += 1

    # Release video capture object
    cap.release()
    print(f"Extracted {saved_count} frames, total frames: {frame_count}")

    return frame_infos


class InferenceDataset:
    def __init__(self, image_paths, testsize):
        self.testsize = testsize
        self.images = image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.size = len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = self.rgb_loader(image_path)
        HH, WW = image.size

        # Apply transform (convert PIL image to tensor)
        image_tensor = self.transform(image)

        return image_tensor, HH, WW, image_path

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


def predict_frames(model, image_paths, testsize, device, constraint_info=None):
    """
    Predict multiple image frames and apply video label constraints
    Args:
        model: Pretrained model
        image_paths: List of image paths
        testsize: Test size
        device: Device
        constraint_info: Constraint info dictionary
    Returns:
        dict: Prediction result dictionary, key is image path, value is (prediction, confidence)
    """
    dataset = InferenceDataset(image_paths, testsize)
    predictions = {}

    model.eval()
    with torch.no_grad():
        total = len(dataset)
        for idx in range(len(dataset)):
            if idx % 10 == 0:
                print(f"Prediction progress: {idx}/{total} ({idx/total*100:.1f}%)")

            image, HH, WW, image_path = dataset[idx]

            # Ensure image is tensor and add batch dimension
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Expected image to be torch.Tensor, got {type(image)}")

            image = image.unsqueeze(0).to(device)

            # Forward pass
            pred = model.forward(image, training=False)

            # Upsample to original size
            pred = F.interpolate(pred, size=[WW, HH], mode="bilinear", align_corners=False)

            # Sigmoid activation
            pred_sigmoid = pred.sigmoid()

            # Apply constraint
            if constraint_info and constraint_info["has_constraint"]:
                pred_sigmoid = apply_video_constraint(pred_sigmoid, constraint_info)

            # Compute more reliable confidence: weighted mean and proportion of pixels above threshold
            pred_np = pred_sigmoid.data.cpu().numpy().squeeze()
            foreground_ratio = (pred_np > 0.6).sum() / pred_np.size
            confidence = 0.85 * pred_np.mean() + 0.15 * foreground_ratio

            # Save prediction and confidence
            predictions[image_path] = (pred_np, confidence)

    return predictions


def apply_video_constraint(pred_sigmoid, constraint_info):
    """
    Adjust prediction according to video label constraint and set corresponding probability value
    Args:
        pred_sigmoid: Prediction tensor
        constraint_info: Constraint info dictionary
    Returns:
        torch.Tensor: Adjusted prediction for subsequent binarization
    """
    if not constraint_info["has_constraint"]:
        return pred_sigmoid

    expected_smoke = constraint_info["expected_smoke"]
    constraint_confidence = constraint_info["constraint_confidence"]
    constraint_strength = constraint_info["constraint_strength"]

    # Use constraint_confidence as the constraint probability value
    if expected_smoke is False:
        # "Definitely no" region: directly use constraint_confidence
        constraint_prob = constraint_confidence
        # Negative samples are still mainly based on the constraint
        pred_sigmoid = pred_sigmoid * 0.4 + constraint_prob * 0.6

    elif expected_smoke is True:
        # "Definitely yes" or "maybe yes" region: directly use constraint_confidence
        constraint_prob = constraint_confidence
        # Positive samples are more biased towards model prediction
        pred_sigmoid = pred_sigmoid * 0.8 + constraint_prob * 0.2

    # Ensure the value is within a valid range
    pred_sigmoid = torch.clamp(pred_sigmoid, 0, 1)

    return pred_sigmoid


def binarize_prediction(pred, threshold=0.6, constraint_info=None):
    """
    Binarize prediction to ensure pseudo label has only 0 and 255 values
    Use unified threshold 0.6, constraint effect is reflected by probability value
    Args:
        pred: Prediction array (numpy array, range 0-1)
        threshold: Unified binarization threshold (default 0.6)
        constraint_info: Constraint info (kept for compatibility, not used to adjust threshold)
    Returns:
        numpy.ndarray: Binarized result (0 or 1)
    """
    # Use a unified threshold for binarization
    # The effect of the constraint has already been reflected by the probability value set in apply_video_constraint:
    # - Strong positive constraint: probability 0.8-0.9 > 0.6 → foreground
    # - Weak positive constraint: probability 0.7 > 0.6 → foreground
    # - Weak negative constraint: probability 0.4 < 0.6 → background
    # - Strong negative constraint: probability 0.1-0.3 < 0.6 → background

    binary_result = (pred > threshold).astype(np.float32)

    return binary_result


def analyze_prediction_distribution(pred, frame_path, constraint_info=None):
    """
    Analyze value distribution of prediction for debugging
    Args:
        pred: Prediction array
        frame_path: Frame path
        constraint_info: Constraint info
    """
    min_val = pred.min()
    max_val = pred.max()
    mean_val = pred.mean()
    std_val = pred.std()

    # Count number of pixels in different value ranges
    low_pixels = np.sum(pred < 0.3)
    mid_pixels = np.sum((pred >= 0.3) & (pred <= 0.7))
    high_pixels = np.sum(pred > 0.7)
    total_pixels = pred.size

    frame_name = os.path.basename(frame_path)
    print(f"  Frame {frame_name}:")
    print(f"    Value range: [{min_val:.3f}, {max_val:.3f}], mean: {mean_val:.3f}, std: {std_val:.3f}")
    print(
        f"    Pixel distribution: low(<0.3):{low_pixels}({low_pixels/total_pixels*100:.1f}%), "
        f"mid(0.3-0.7):{mid_pixels}({mid_pixels/total_pixels*100:.1f}%), "
        f"high(>0.7):{high_pixels}({high_pixels/total_pixels*100:.1f}%)"
    )

    if constraint_info and constraint_info["has_constraint"]:
        print(f"    Constraint: {constraint_info['constraint_strength']}, expected smoke: {constraint_info['expected_smoke']}")


def calculate_mask_quality(pred_tensor):
    """
    Calculate quality score of predicted mask
    Args:
        pred_tensor: Prediction tensor (torch.Tensor)
    Returns:
        float: Quality score, lower means better (smoother)
    """
    # Ensure input is 4D tensor (batch_size, channels, height, width)
    if len(pred_tensor.shape) == 2:
        pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
    elif len(pred_tensor.shape) == 3:
        pred_tensor = pred_tensor.unsqueeze(0)

    # Compute gradient magnitude
    grad_x = gradient_x(pred_tensor)
    grad_y = gradient_y(pred_tensor)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    # Compute Laplacian edge
    lap_edge = torch.abs(laplacian_edge(pred_tensor))

    # Compute quality metrics (lower is better)
    # 1. Mean gradient change (edge smoothness)
    gradient_score = gradient_magnitude.mean().item()

    # 2. Mean Laplacian response (overall smoothness)
    laplacian_score = lap_edge.mean().item()

    # 3. Std of prediction values (consistency)
    consistency_score = pred_tensor.std().item()

    # Overall quality score (weighted average)
    quality_score = gradient_score * 0.4 + laplacian_score * 0.4 + consistency_score * 0.2

    return quality_score


# --- High-confidence frame selection (with similarity constraint) ---
def select_high_confidence_frames_with_similarity(predictions, frame_infos, context_frames=2, threshold=0.5):
    """
    Select high-confidence frames and their context frames, combine mask similarity between frames to improve temporal consistency of pseudo labels.
    Args:
        predictions: Prediction result dictionary
        frame_infos: Frame info dictionary
        context_frames: Number of context frames
        threshold: Confidence threshold
    Returns:
        list: List of selected frame paths
    """
    # 1. Sort by confidence, select top-k as candidate center frames
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1][1], reverse=True)
    if not sorted_preds:
        print("Warning: No available frames")
        return []
    top_k = min(3, len(sorted_preds))
    candidate_centers = [frame_path for frame_path, _ in sorted_preds[:top_k]]

    best_group = None
    best_score = -float("inf")
    best_center = None
    best_conf = 0.0
    best_sim = 0.0

    for center in candidate_centers:
        center_id = frame_infos[center]
        group = [center]
        # Add context frames before and after in the group
        for i in range(1, context_frames + 1):
            prev_id = center_id - i
            next_id = center_id + i
            prevs = [p for p, id in frame_infos.items() if id == prev_id]
            nexts = [p for p, id in frame_infos.items() if id == next_id]
            if prevs:
                group.append(prevs[0])
            if nexts:
                group.append(nexts[0])
        # Only consider frames that actually exist
        group = [f for f in group if f in predictions]
        if len(group) < 2:
            continue
        # Compute mean mask similarity within group
        sims = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sim = calculate_frame_similarity(predictions[group[i]][0], predictions[group[j]][0])
                sims.append(sim)
        avg_sim = np.mean(sims) if sims else 0.0
        # Center frame confidence
        conf = predictions[center][1]
        # Overall score: similarity + confidence
        score = avg_sim * 0.2 + conf * 0.8
        print(f"  Candidate center frame {os.path.basename(center)}: confidence={conf:.3f}, avg similarity={avg_sim:.3f}, overall score={score:.3f}")
        if score > best_score:
            best_score = score
            best_group = group
            best_center = center
            best_conf = conf
            best_sim = avg_sim

    if not best_group:
        print("Warning: No valid high-confidence group found, fallback to highest confidence frame")
        # Fallback to original logic
        fallback = sorted_preds[0][0]
        fallback_id = frame_infos[fallback]
        fallback_group = set([fallback])
        for i in range(1, context_frames + 1):
            prev_id = fallback_id - i
            next_id = fallback_id + i
            prevs = [p for p, id in frame_infos.items() if id == prev_id]
            nexts = [p for p, id in frame_infos.items() if id == next_id]
            if prevs:
                fallback_group.add(prevs[0])
            if nexts:
                fallback_group.add(nexts[0])
        print(f"Selected highest confidence frame {os.path.basename(fallback)} and its context frames, total {len(fallback_group)} frames")
        return list(fallback_group)

    center_name = os.path.basename(best_center) if best_center is not None else "<None>"
    print(f"Selected best group center frame: {center_name} (confidence={best_conf:.3f}, avg similarity={best_sim:.3f})")
    print(f"Selected best group center frame and its context frames, total {len(set(best_group))} frames")
    return list(set(best_group))


# Replace the original interface name to ensure compatibility
select_high_confidence_frames = select_high_confidence_frames_with_similarity


def save_results(predictions, selected_frames, video_name, output_path, start_idx=0, constraint_info=None):
    """
    Save selected frames and pseudo labels (ensure pseudo labels are pure binary)
    Args:
        predictions: Prediction result dictionary, key is image path, value is (prediction, confidence)
        selected_frames: List of selected frame paths
        video_name: Video name, used for file naming
        output_path: Output root path
        start_idx: File index start value, to ensure unique index across multiple videos
        constraint_info: Constraint info, used for binarization threshold adjustment

    Output structure:
    output_path/
        ├── img/   # Store original images
        └── gt/    # Store pseudo labels (pure binary: 0 and 255)
    """
    img_dir = os.path.join(output_path, "img")
    gt_dir = os.path.join(output_path, "gt")

    # Create output directories
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Record saved image and label info
    saved_count = 0
    saved_files = []  # Used to track saved file names (without extension)

    print(f"Saving binarized pseudo labels for {len(selected_frames)} frames ...")

    for idx, frame_path in enumerate(selected_frames, start=start_idx):
        # Get original image and prediction
        pred, confidence = predictions[frame_path]

        # Analyze prediction distribution (for debugging)
        if idx == start_idx:  # Only analyze the first frame to avoid excessive output
            analyze_prediction_distribution(pred, frame_path, constraint_info)

        # Binarize prediction (use unified threshold 0.6)
        binary_pred = binarize_prediction(pred, threshold=0.6)

        # Generate new file name: video name + index
        new_filename = f"{video_name}_{idx:02d}"
        output_image_path = os.path.join(img_dir, f"{new_filename}.jpg")
        output_label_path = os.path.join(gt_dir, f"{new_filename}.png")

        # Use PIL to read and save image, keep original quality
        with Image.open(frame_path) as img:
            img.save(output_image_path)

        # Save binarized pseudo label (only 0 and 255 values)
        binary_label = (binary_pred * 255).astype(np.uint8)
        cv2.imwrite(output_label_path, binary_label)

        # Verify that the saved label is indeed binary
        unique_values = np.unique(binary_label)
        if len(unique_values) > 2 or (len(unique_values) == 2 and not (0 in unique_values and 255 in unique_values)):
            print(f"Warning: Frame {new_filename} pseudo label is not pure binary! Unique values: {unique_values}")
        elif idx == start_idx:  # Only print validation info for the first frame
            print(f"    Binarization check: unique gray values {unique_values} ✓")

        # Add to saved file list (without extension, but with full new file name)
        saved_files.append(new_filename)
        saved_count += 1

    print(f"Saved {saved_count} pairs of images and binarized pseudo labels")
    print(f"  - Image directory: {img_dir}")
    print(f"  - Label directory: {gt_dir}")
    print(f"  - All pseudo labels are pure binary images (0 and 255)")

    return saved_files, start_idx + saved_count  # Return the list of saved file names and the next starting index


def generate_transmission_maps(output_path, saved_files):
    """
    Generate transmission maps for saved images
    Args:
        output_path: Output root path
        saved_files: List of saved file names

    Output structure:
    output_path/
        ├── img/   # Store original images
        ├── gt/    # Store pseudo labels
        └── trans/ # Store transmission maps
    """
    img_dir = os.path.join(output_path, "img")
    trans_dir = os.path.join(output_path, "trans")

    # Create transmission map output directory
    os.makedirs(trans_dir, exist_ok=True)

    print(f"Generating transmission maps for {len(saved_files)} images ...")
    total = len(saved_files)
    for i, filename in enumerate(saved_files):
        if i % 5 == 0:
            print(f"Transmission map generation progress: {i}/{total} ({i/total*100:.1f}%)")

        # Load original image
        img_path = os.path.join(img_dir, f"{filename}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{filename}.png")
            if not os.path.exists(img_path):
                print(f"Warning: Cannot find image file {filename}")
                continue

        # Read image
        src = cv2.imread(img_path)
        if src is None:
            print(f"Warning: Cannot read image {img_path}")
            continue

        # Generate transmission map - use function from transmission_map.py
        trans = find_transmission_map(src)

        # Save transmission map
        trans_path = os.path.join(trans_dir, f"{filename}.png")
        cv2.imwrite(trans_path, trans * 255)

    print(f"Transmission map generation completed, saved in: {trans_dir}")


def clean_output_directories(output_path):
    """
    Clean all existing files in output directories
    Args:
        output_path: Output root path
    """
    directories = ["img", "gt", "trans"]

    for directory in directories:
        dir_path = os.path.join(output_path, directory)
        if os.path.exists(dir_path):
            print(f"Cleaning directory: {dir_path}")
            # Delete all files in directory
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Directory {directory} cleared")
        else:
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

    print("All output directories cleaned")


def validate_pseudo_labels(output_path):
    """
    Validate whether generated pseudo labels are pure binary images
    Args:
        output_path: Output path
    """
    gt_dir = os.path.join(output_path, "gt")
    if not os.path.exists(gt_dir):
        print("Warning: Pseudo label directory does not exist")
        return

    label_files = [f for f in os.listdir(gt_dir) if f.endswith(".png")]
    if not label_files:
        print("Warning: No pseudo label files found")
        return

    print(f"\nValidating {len(label_files)} pseudo label files ...")

    non_binary_count = 0
    gray_value_stats = {}

    for label_file in label_files:
        label_path = os.path.join(gt_dir, label_file)
        label = cv2.imread(label_path, 0)

        if label is None:
            print(f"Warning: Cannot read label file {label_file}")
            continue

        unique_values = np.unique(label)

        # Record gray value statistics
        for val in unique_values:
            gray_value_stats[val] = gray_value_stats.get(val, 0) + 1

        # Check if it is pure binary
        if len(unique_values) > 2 or (len(unique_values) == 2 and not (0 in unique_values and 255 in unique_values)):
            non_binary_count += 1
            if non_binary_count <= 5:  # Only show first 5 non-binary files
                print(f"  Non-binary label: {label_file}, gray values: {unique_values}")

    print(f"\nValidation result:")
    print(f"  Total files: {len(label_files)}")
    print(f"  Binary files: {len(label_files) - non_binary_count}")
    print(f"  Non-binary files: {non_binary_count}")

    print(f"\nAll gray value distribution:")
    for gray_val, count in sorted(gray_value_stats.items()):
        print(f"  Gray value {gray_val}: {count} occurrences")

    if non_binary_count == 0:
        print("All pseudo labels are pure binary images (only 0 and 255)")
    else:
        print(f"{non_binary_count} pseudo labels are not pure binary images")

    return non_binary_count == 0


def calculate_frame_similarity(pred1, pred2):
    """
    Calculate similarity between two prediction results
    Args:
        pred1: First prediction (numpy array)
        pred2: Second prediction (numpy array)
    Returns:
        float: Similarity score (0-1, higher means more similar)
    """
    if pred1.shape != pred2.shape:
        return 0.0

    # Compute IoU similarity
    pred1_binary = (pred1 > 0.5).astype(np.float32)
    pred2_binary = (pred2 > 0.5).astype(np.float32)

    intersection = np.sum(pred1_binary * pred2_binary)
    union = np.sum(np.maximum(pred1_binary, pred2_binary))

    if union == 0:
        return 1.0  # Both are all black, considered similar

    iou = intersection / union

    # Also compute pixel-level correlation
    correlation = np.corrcoef(pred1.flatten(), pred2.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    # Combine IoU and correlation
    similarity = 0.7 * iou + 0.3 * max(0, correlation)

    return similarity


def main():
    """
    Main function - execute the whole pseudo label generation process:
    1. Load pretrained model
    2. Load video label constraints (if enabled)
    3. Process all video files
    4. Extract frames from each video
    5. Predict on frames (apply constraints)
    6. Select high-confidence frames and their context
    7. Output original images and pseudo labels to corresponding constraint directory
    8. Generate transmission maps
    """
    opt = arg_parse()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
    generator.load_state_dict(torch.load(opt.pretrained_weights, map_location=device, weights_only=False))
    generator.to(device)
    generator.eval()
    print(f"Model loaded: {opt.pretrained_weights}")

    # Load video labels (if constraints enabled)
    video_labels = {}
    if opt.constraint_type != "none":
        video_labels = load_video_labels(opt.video_labels_csv)
        print(f"Constraint type: {opt.constraint_type}")

    # Set output path according to constraint type
    if opt.constraint_type == "none":
        final_output_path = os.path.join(opt.output_path, "non_constraint")
    elif opt.constraint_type == "citizen":
        final_output_path = os.path.join(opt.output_path, "citizen_constraint")
    elif opt.constraint_type == "expert":
        final_output_path = os.path.join(opt.output_path, "expert_constraint")
    else:
        final_output_path = opt.output_path

    # Create output directory
    os.makedirs(final_output_path, exist_ok=True)
    print(f"Output path: {final_output_path}")

    # Clean existing files in output directory
    clean_output_directories(final_output_path)

    # Get all video files
    video_files = glob.glob(os.path.join(opt.videos_path, "*.mp4")) + glob.glob(os.path.join(opt.videos_path, "*.avi"))
    print(f"Found {len(video_files)} video files")

    if len(video_files) == 0:
        print(f"Error: No video files found in {opt.videos_path}")
        return

    # Used to ensure unique index across multiple videos
    next_idx = 0
    all_saved_files = []
    processed_videos = 0
    skipped_videos = 0

    for video_file in video_files:
        video_name = os.path.basename(video_file).split(".")[0]  # Get full video name (with extension)
        print(f"\nProcessing video: {video_name}")

        # Get video constraint info (now using full video name)
        constraint_info = get_video_constraint_info(video_name, video_labels, opt.constraint_type)

        if constraint_info["has_constraint"]:
            print(
                f"  Constraint info: {constraint_info['constraint_strength']}, "
                f"expected smoke: {constraint_info['expected_smoke']}, "
                f"constraint confidence: {constraint_info['constraint_confidence']:.2f}"
            )

            # New: For negative samples, skip prediction, sample 5 frames and save all-zero masks
            if constraint_info["expected_smoke"] is False:
                print(f"  Skip model prediction, negative sample: sample 5 frames and save all-zero masks")
                temp_frames_dir = os.path.join(final_output_path, f"temp_{video_name}")
                os.makedirs(temp_frames_dir, exist_ok=True)
                frame_infos = extract_frames_from_video(video_file, temp_frames_dir, opt.sampling_rate)
                frame_paths = list(frame_infos.keys())
                if len(frame_paths) == 0:
                    print(f"Skip video {video_name}: cannot extract frames")
                    skipped_videos += 1
                    continue
                # Randomly sample 5 frames
                import random

                sample_num = min(5, len(frame_paths))
                sampled_frames = random.sample(frame_paths, sample_num)
                # Save original images and all-zero masks
                img_dir = os.path.join(final_output_path, "img")
                gt_dir = os.path.join(final_output_path, "gt")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(gt_dir, exist_ok=True)
                for idx, frame_path in enumerate(sampled_frames, start=next_idx):
                    new_filename = f"{video_name}_{idx:02d}"
                    output_image_path = os.path.join(img_dir, f"{new_filename}.jpg")
                    output_label_path = os.path.join(gt_dir, f"{new_filename}.png")
                    with Image.open(frame_path) as img:
                        img.save(output_image_path)
                    # Generate all-zero mask, same size as original image
                    img0 = Image.open(frame_path)
                    mask = np.zeros(img0.size[::-1], dtype=np.uint8)  # (H, W)
                    cv2.imwrite(output_label_path, mask)
                next_idx += sample_num
                all_saved_files.extend([f"{video_name}_{i:02d}" for i in range(next_idx - sample_num, next_idx)])
                processed_videos += 1
                # Clean up temporary frame directory
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                os.rmdir(temp_frames_dir)
                print(f"Video {video_name} processed (negative sample all-zero mask)")
                continue
        else:
            print(f"  No constraint info, normal processing")

        # Create temporary frame directory for each video
        temp_frames_dir = os.path.join(final_output_path, f"temp_{video_name}")
        os.makedirs(temp_frames_dir, exist_ok=True)

        # Extract frames
        frame_infos = extract_frames_from_video(video_file, temp_frames_dir, opt.sampling_rate)
        if not frame_infos:
            print(f"Skip video {video_name}: cannot extract frames")
            skipped_videos += 1
            continue

        # Get all extracted frame paths
        frame_paths = list(frame_infos.keys())

        # Predict all frames (apply constraints)
        predictions = predict_frames(generator, frame_paths, opt.testsize, device, constraint_info)

        # Select high-confidence frames and their context (no longer includes temporal consistency constraint)
        selected_frames = select_high_confidence_frames(predictions, frame_infos, opt.context_frames, opt.threshold)

        print(f"Selected {len(selected_frames)} frames (from {len(frame_paths)} frames)")

        # Save results to corresponding constraint directory and get next start index
        saved_files, next_idx = save_results(predictions, selected_frames, video_name, final_output_path, next_idx, constraint_info)
        all_saved_files.extend(saved_files)
        processed_videos += 1

        # Clean up temporary frame directory
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir(temp_frames_dir)

        print(f"Video {video_name} processed")

    # After all videos are processed, generate all transmission maps at once
    if all_saved_files:
        generate_transmission_maps(final_output_path, all_saved_files)

    # Validate whether all generated pseudo labels are pure binary images
    is_all_binary = validate_pseudo_labels(final_output_path)

    print(f"\nProcessing summary:")
    print(f"  Processed videos: {processed_videos}")
    print(f"  Skipped videos: {skipped_videos}")
    print(f"  Constraint type: {opt.constraint_type}")
    print(f"  Output directory: {final_output_path}")
    print(f"  Pseudo label quality: {' All pure binary' if is_all_binary else '️ Non-binary labels exist'}")
    print("Directory structure:")
    print("  - img/    (original images)")
    print("  - gt/     (pseudo labels - pure binary: 0 and 255)")
    print("  - trans/  (transmission maps)")

    if opt.constraint_type != "none":
        print(f"\nConstraint effect explanation (unified threshold 0.6):")
        print(f"  • 'Gold positive' constraint: probability 0.9 > 0.6 → foreground (255)")
        print(f"  • 'Strong positive' constraint: probability 0.8 > 0.6 → foreground (255)")
        print(f"  • 'Weak positive/maybe' constraint: probability 0.7 > 0.6 → foreground (255)")
        print(f"  • 'Other positive' constraint: probability 0.65 > 0.6 → foreground (255)")
        print(f"  • 'Other negative' constraint: probability 0.58 < 0.6 → background (0)")
        print(f"  • 'Weak negative' constraint: probability 0.55 < 0.6 → background (0) (but close to threshold, give original prediction a chance)")
        print(f"  • 'Strong negative' constraint: probability 0.4 < 0.6 → background (0)")
        print(f"  • 'Gold negative' constraint: probability 0.2 < 0.6 → background (0)")
        print(f"  • Weak constraint keeps 40% original prediction + 60% constraint probability, balancing constraint and model prediction")
        print(f"  • Unified threshold 0.6 ensures consistency of binarization decision")


if __name__ == "__main__":
    main()
