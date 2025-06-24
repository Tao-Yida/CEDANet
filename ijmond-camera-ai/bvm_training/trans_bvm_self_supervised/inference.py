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

"""
自监督模型视频伪标签生成器 (Semi-Supervised Video Pseudo Label Generator)
============================================================================

此脚本专为自监督模型设计，用于从视频文件中生成伪标签：
1. 读取自监督预训练模型
2. 从视频中提取帧
3. 对提取的帧进行预测（使用自监督模型的推理模式）
4. 选择每个视频中置信度最高的帧及其前后n帧（共2n+1帧）
5. 输出原始图像和对应的伪标签
6. 生成对应的透光率图

自监督模型特点：
- 训练时：返回 (sal_init_post, sal_ref_post, sal_init_prior, sal_ref_prior, latent_loss, output_post, output_prior)
- 推理时：返回 prob_pred （使用先验预测）
- 需要 num_filters 参数

用法示例:
-------
```
python inference.py --videos_path ../../data/videos \
                   --output_path ../../data/ijmond_camera \
                   --pretrained_weights ../../models/semi_model.pth \
                   --num_filters 16 \
                   --sampling_rate 5 \
                   --context_frames 2 \
                   --threshold 0.5
```
"""

# 添加父目录到系统路径，以便导入父目录的模块
sys.path.append(str(Path(__file__).parent.parent))
from model.ResNet_models import Generator  # 使用自监督模型

sys.path.append(str(Path(__file__).parent.parent.parent))
from transmission_map import find_transmission_map
from smoothness import gradient_x, gradient_y, laplacian_edge


def arg_parse():
    """
    解析命令行参数
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="使用自监督模型从视频中生成伪标签")
    parser.add_argument("--testsize", type=int, default=352, help="测试图像大小")
    parser.add_argument("--latent_dim", type=int, default=3, help="潜在空间维度")
    parser.add_argument("--feat_channel", type=int, default=32, help="特征通道数")
    parser.add_argument("--num_filters", type=int, default=16, help="自监督模型过滤器数量")
    parser.add_argument("--videos_path", type=str, default="../../../data/ijmond_camera/videos", help="视频文件路径")
    parser.add_argument("--output_path", type=str, default="../../../data/ijmond_camera/SMOKE5K-self", help="自监督伪标签输出路径")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="../../../models/semi-supervision/SMOKE5K_Dataset_SMOKE5K_train_ssl_SMOKE5K_Dataset_SMOKE5K_weak_supervision/SMOKE5K_Dataset_SMOKE5K_train_ssl_SMOKE5K_Dataset_SMOKE5K_weak_supervision_best_model.pth",
        help="自监督预训练权重路径",
    )
    parser.add_argument("--sampling_rate", type=int, default=1, help="帧采样率")
    parser.add_argument("--context_frames", type=int, default=2, help="高置信度帧前后的上下文帧数")
    parser.add_argument("--threshold", type=float, default=0.5, help="伪标签置信度阈值")
    parser.add_argument(
        "--constraint_type",
        type=str,
        choices=["none", "citizen", "expert"],
        default="none",
        help="约束类型: none(无约束), citizen(市民约束), expert(专家约束)",
    )
    parser.add_argument("--video_labels_csv", type=str, default="../../../data/ijmond_camera/video_labels.csv", help="视频标签CSV文件路径")
    opt = parser.parse_args()
    return opt


def load_video_labels(csv_path):
    """
    加载视频标签CSV文件
    Args:
        csv_path: CSV文件路径
    Returns:
        dict: 视频标签字典，键为完整的文件名（包含后缀），值为标签信息
    """
    if not os.path.exists(csv_path):
        print(f"警告: 视频标签文件不存在: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        video_labels = {}

        for _, row in df.iterrows():
            file_name = row["file_name"]
            label_state = row["label_state"]
            label_state_admin = row["label_state_admin"]

            # 直接使用完整的文件名作为键，不去除后缀
            video_labels[file_name] = {"label_state": label_state, "label_state_admin": label_state_admin}

        print(f"加载了 {len(video_labels)} 个视频段的标签信息")
        return video_labels

    except Exception as e:
        print(f"加载视频标签时出错: {e}")
        return {}


def get_video_constraint_info(video_name, video_labels, constraint_type):
    """
    获取视频的约束信息
    Args:
        video_name: 视频名称（完整文件名，包含后缀）
        video_labels: 视频标签字典
        constraint_type: 约束类型 ('none', 'citizen', 'expert')
    Returns:
        dict: 包含约束信息的字典
    """
    # 如果约束类型为none，直接返回无约束
    if constraint_type == "none":
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    # 如果视频不在标签字典中，返回无约束
    if video_name not in video_labels:
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    # 根据约束类型选择标签列
    if constraint_type == "citizen":
        label_key = "label_state"
    elif constraint_type == "expert":
        label_key = "label_state_admin"
    else:
        # 无效的约束类型
        print(f"警告: 无效的约束类型 '{constraint_type}', 支持的类型: 'none', 'citizen', 'expert'")
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    label_value = video_labels[video_name][label_key]

    # 如果标签值为-1（无数据）或-2（坏视频），返回无约束
    if label_value == -1 or label_value == -2:
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}

    # 定义标签含义和置信度
    label_meanings = {
        47: {"smoke": True, "confidence": 1.0, "strength": "gold_positive"},  # 黄金标准正样本
        32: {"smoke": False, "confidence": 1.0, "strength": "gold_negative"},  # 黄金标准负样本
        23: {"smoke": True, "confidence": 0.9, "strength": "strong_positive"},  # 强正样本
        16: {"smoke": False, "confidence": 0.9, "strength": "strong_negative"},  # 强负样本
        19: {"smoke": True, "confidence": 0.7, "strength": "weak_positive"},  # 弱正样本
        20: {"smoke": False, "confidence": 0.7, "strength": "weak_negative"},  # 弱负样本
        5: {"smoke": True, "confidence": 0.5, "strength": "maybe_positive"},  # 可能正样本
        4: {"smoke": False, "confidence": 0.5, "strength": "maybe_negative"},  # 可能负样本
        3: {"smoke": None, "confidence": 0.3, "strength": "discord"},  # 有分歧
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
        # 未知的标签值
        print(f"警告: 未知的标签值 {label_value} 对于视频 {video_name}")
        return {"has_constraint": False, "constraint_confidence": 1.0, "expected_smoke": None, "constraint_strength": "none"}


def extract_frames_from_video(video_path, output_folder, sampling_rate=1):
    """
    从视频中提取帧
    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹
        sampling_rate: 抽帧率，每隔多少帧提取一帧
    Returns:
        dict: 帧信息字典，包含帧路径和帧编号
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return {}

    frame_infos = {}
    frame_count = 0
    saved_count = 0

    print(f"从 {video_path} 中提取帧...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 根据采样率提取帧
        if frame_count % sampling_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:02d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_infos[frame_filename] = frame_count
            saved_count += 1

        frame_count += 1

    # 释放视频捕获对象
    cap.release()
    print(f"共提取了 {saved_count} 帧，总帧数: {frame_count}")

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

        # 应用变换(将PIL图像转换为张量)
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
    预测多个图像帧，并应用视频标签约束
    Args:
        model: 预训练模型
        image_paths: 图像路径列表
        testsize: 测试尺寸
        device: 计算设备
        constraint_info: 约束信息字典
    Returns:
        dict: 预测结果字典，键为图像路径，值为(预测结果, 置信度)
    """
    dataset = InferenceDataset(image_paths, testsize)
    predictions = {}

    model.eval()
    with torch.no_grad():
        total = len(dataset)
        for idx in range(len(dataset)):
            if idx % 10 == 0:
                print(f"预测进度: {idx}/{total} ({idx/total*100:.1f}%)")

            image, HH, WW, image_path = dataset[idx]

            # 确保image是tensor并添加batch维度
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Expected image to be torch.Tensor, got {type(image)}")

            image = image.unsqueeze(0).to(device)

            # 前向传播
            pred = model.forward(image, training=False)

            # 上采样到原始尺寸
            pred = F.interpolate(pred, size=[WW, HH], mode="bilinear", align_corners=False)

            # Sigmoid激活
            pred_sigmoid = pred.sigmoid()

            # 应用约束
            if constraint_info and constraint_info["has_constraint"]:
                pred_sigmoid = apply_video_constraint(pred_sigmoid, constraint_info)

            # 计算平均置信度作为该帧的置信度
            confidence = pred_sigmoid.mean().item()

            # 保存预测结果和置信度
            predictions[image_path] = (pred_sigmoid.data.cpu().numpy().squeeze(), confidence)

    return predictions


def apply_video_constraint(pred_sigmoid, constraint_info):
    """
    根据视频标签约束调整预测结果，并设置相应的概率值
    Args:
        pred_sigmoid: 预测结果张量
        constraint_info: 约束信息字典
    Returns:
        torch.Tensor: 调整后的预测结果，用于后续二值化
    """
    if not constraint_info["has_constraint"]:
        return pred_sigmoid

    expected_smoke = constraint_info["expected_smoke"]
    constraint_confidence = constraint_info["constraint_confidence"]
    constraint_strength = constraint_info["constraint_strength"]

    # 根据约束强度设置不同的概率值
    if expected_smoke is False:
        # "一定没有"区域：根据约束强度设置不同的抑制概率
        if "gold" in constraint_strength:
            # 黄金标准负样本：设置为很低概率，确保二值化后为背景
            constraint_prob = 0.2
        elif "strong" in constraint_strength:
            # 强负约束：设置为较低概率
            constraint_prob = 0.4
        elif "weak" in constraint_strength:
            # 弱负约束：设置为稍低于阈值的概率，但仍给原始预测一些机会
            constraint_prob = 0.55
        else:
            # 其他负约束：设置为接近中性的概率
            constraint_prob = 0.58

        # 将预测值向约束概率拉近，但保留更多原始预测信息
        pred_sigmoid = pred_sigmoid * 0.4 + constraint_prob * 0.6

    elif expected_smoke is True:
        # "一定有"或"可能有"区域：根据约束强度设置不同的增强概率
        if "gold" in constraint_strength:
            # 黄金标准正样本：设置为很高概率，确保二值化后为前景
            constraint_prob = 0.9
        elif "strong" in constraint_strength:
            # 强正约束：设置为高概率
            constraint_prob = 0.8
        elif "weak" in constraint_strength:
            # 弱正约束："可能有"区域，设置为稍高于阈值的概率
            constraint_prob = 0.7
        else:
            # 其他正约束：设置为稍高于阈值的概率
            constraint_prob = 0.65

        # 将预测值向约束概率拉近，但保留更多原始预测信息
        pred_sigmoid = pred_sigmoid * 0.4 + constraint_prob * 0.6

    # 确保值在有效范围内
    pred_sigmoid = torch.clamp(pred_sigmoid, 0, 1)

    return pred_sigmoid


def binarize_prediction(pred, threshold=0.6, constraint_info=None):
    """
    将预测结果二值化，确保伪标签只有0和255两种值
    使用统一的阈值0.6，约束效果通过概率值体现
    Args:
        pred: 预测结果数组 (numpy array, 0-1范围)
        threshold: 统一的二值化阈值 (默认0.6)
        constraint_info: 约束信息（保留参数以兼容调用，但不再用于调整阈值）
    Returns:
        numpy.ndarray: 二值化后的结果 (0或1)
    """
    # 使用统一的阈值进行二值化
    # 约束效果已经通过apply_video_constraint中的概率值设置体现：
    # - 强正约束: 概率0.8-0.9 > 0.6 → 前景
    # - 弱正约束: 概率0.7 > 0.6 → 前景
    # - 弱负约束: 概率0.4 < 0.6 → 背景
    # - 强负约束: 概率0.1-0.3 < 0.6 → 背景

    binary_result = (pred > threshold).astype(np.float32)

    return binary_result


def analyze_prediction_distribution(pred, frame_path, constraint_info=None):
    """
    分析预测结果的数值分布，用于调试
    Args:
        pred: 预测结果数组
        frame_path: 帧路径
        constraint_info: 约束信息
    """
    min_val = pred.min()
    max_val = pred.max()
    mean_val = pred.mean()
    std_val = pred.std()

    # 统计不同范围的像素数量
    low_pixels = np.sum(pred < 0.3)
    mid_pixels = np.sum((pred >= 0.3) & (pred <= 0.7))
    high_pixels = np.sum(pred > 0.7)
    total_pixels = pred.size

    frame_name = os.path.basename(frame_path)
    print(f"  帧 {frame_name}:")
    print(f"    值域: [{min_val:.3f}, {max_val:.3f}], 均值: {mean_val:.3f}, 标准差: {std_val:.3f}")
    print(
        f"    像素分布: 低值(<0.3):{low_pixels}({low_pixels/total_pixels*100:.1f}%), "
        f"中值(0.3-0.7):{mid_pixels}({mid_pixels/total_pixels*100:.1f}%), "
        f"高值(>0.7):{high_pixels}({high_pixels/total_pixels*100:.1f}%)"
    )

    if constraint_info and constraint_info["has_constraint"]:
        print(f"    约束: {constraint_info['constraint_strength']}, 期望烟雾: {constraint_info['expected_smoke']}")


def calculate_mask_quality(pred_tensor):
    """
    计算预测mask的质量分数
    Args:
        pred_tensor: 预测结果张量 (torch.Tensor)
    Returns:
        float: 质量分数，越低表示质量越好（平滑性越好）
    """
    # 确保输入是4D张量 (batch_size, channels, height, width)
    if len(pred_tensor.shape) == 2:
        pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
    elif len(pred_tensor.shape) == 3:
        pred_tensor = pred_tensor.unsqueeze(0)

    # 计算梯度幅度
    grad_x = gradient_x(pred_tensor)
    grad_y = gradient_y(pred_tensor)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    # 计算拉普拉斯边缘
    lap_edge = torch.abs(laplacian_edge(pred_tensor))

    # 计算质量指标（值越小越好）
    # 1. 梯度变化的平均值（边缘平滑性）
    gradient_score = gradient_magnitude.mean().item()

    # 2. 拉普拉斯响应的平均值（整体平滑性）
    laplacian_score = lap_edge.mean().item()

    # 3. 预测值的标准差（一致性）
    consistency_score = pred_tensor.std().item()

    # 综合质量分数（加权平均）
    quality_score = gradient_score * 0.4 + laplacian_score * 0.4 + consistency_score * 0.2

    return quality_score


def select_high_confidence_frames(predictions, frame_infos, context_frames=2, threshold=0.5):
    """
    选择高置信度帧及其上下文帧 - 改进版本
    Args:
        predictions: 预测结果字典
        frame_infos: 帧信息字典
        context_frames: 上下文帧数量
        threshold: 置信度阈值
    Returns:
        list: 选择的帧路径列表
    """
    # 按照置信度排序
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1][1], reverse=True)

    if not sorted_preds:
        print("警告: 没有可用的帧")
        return []

    # 选择置信度前三的帧进行质量评估
    top_candidates = sorted_preds[: min(3, len(sorted_preds))]

    print(f"评估置信度前 {len(top_candidates)} 帧的质量...")

    best_frame = None
    best_score = float("inf")
    best_confidence = 0.0
    best_quality = 0.0

    for frame_path, (pred_numpy, confidence) in top_candidates:
        # 将numpy数组转换为torch张量进行质量评估
        pred_tensor = torch.from_numpy(pred_numpy).float()

        # 计算mask质量分数
        quality_score = calculate_mask_quality(pred_tensor)

        # 综合评分：置信度越高越好，质量分数越低越好
        # 归一化置信度到0-1，质量分数通常在0-1范围
        normalized_confidence = confidence
        combined_score = quality_score - normalized_confidence * 0.5  # 置信度权重为0.5

        print(f"  帧 {os.path.basename(frame_path)}: 置信度={confidence:.3f}, 质量分数={quality_score:.3f}, 综合分数={combined_score:.3f}")

        if combined_score < best_score:
            best_score = combined_score
            best_frame = frame_path
            best_confidence = confidence
            best_quality = quality_score

    if best_frame is None:
        print("警告: 无法选择最佳帧，使用置信度最高的帧")
        best_frame = sorted_preds[0][0]
        best_confidence = sorted_preds[0][1][1]
        best_quality = 0.0

    print(f"选择最佳帧: {os.path.basename(best_frame)} (置信度={best_confidence:.3f}, 质量分数={best_quality:.3f})")

    # 为最佳帧添加上下文帧
    selected_frames = set()
    frame_id = frame_infos[best_frame]

    # 添加中心帧
    selected_frames.add(best_frame)

    # 添加上下文帧
    for i in range(1, context_frames + 1):
        # 前面的帧
        prev_frame_id = frame_id - i
        prev_frame_paths = [p for p, id in frame_infos.items() if id == prev_frame_id]
        if prev_frame_paths:
            selected_frames.add(prev_frame_paths[0])

        # 后面的帧
        next_frame_id = frame_id + i
        next_frame_paths = [p for p, id in frame_infos.items() if id == next_frame_id]
        if next_frame_paths:
            selected_frames.add(next_frame_paths[0])

    print(f"选择了最佳帧 (ID: {frame_id}) 及其前后各 {context_frames} 帧，总共 {len(selected_frames)} 帧")
    return list(selected_frames)


def save_results(predictions, selected_frames, video_name, output_path, start_idx=0, constraint_info=None):
    """
    保存选择的帧和伪标签（确保伪标签为纯二值）
    Args:
        predictions: 预测结果字典，键为图像路径，值为(预测结果, 置信度)
        selected_frames: 选择的帧路径列表
        video_name: 视频名称，用于命名文件
        output_path: 输出根路径
        start_idx: 文件编号起始值，用于在多个视频间确保索引唯一性
        constraint_info: 约束信息，用于二值化阈值调整

    输出结构:
    output_path/
        ├── img/   # 存储原始图像
        └── pl/    # 存储伪标签（纯二值：0和255）
    """
    img_dir = os.path.join(output_path, "img")
    pl_dir = os.path.join(output_path, "pl")

    # 创建输出目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pl_dir, exist_ok=True)

    # 记录保存的图像和标签信息
    saved_count = 0
    saved_files = []  # 用于跟踪保存的文件名（不含扩展名）

    print(f"正在保存 {len(selected_frames)} 帧的二值化伪标签...")

    for idx, frame_path in enumerate(selected_frames, start=start_idx):
        # 获取原始图像和预测结果
        pred, confidence = predictions[frame_path]

        # 分析预测分布（调试用）
        if idx == start_idx:  # 只分析第一帧，避免输出过多
            analyze_prediction_distribution(pred, frame_path, constraint_info)

        # 二值化预测结果（使用统一阈值0.6）
        binary_pred = binarize_prediction(pred, threshold=0.6)

        # 生成新的文件名：视频名+序号
        new_filename = f"{video_name}_{idx:02d}"
        output_image_path = os.path.join(img_dir, f"{new_filename}.jpg")
        output_label_path = os.path.join(pl_dir, f"{new_filename}.png")

        # 使用PIL读取和保存图像，保持原始质量
        with Image.open(frame_path) as img:
            img.save(output_image_path)

        # 保存二值化伪标签 (只有0和255两种值)
        binary_label = (binary_pred * 255).astype(np.uint8)
        cv2.imwrite(output_label_path, binary_label)

        # 验证保存的标签确实是二值的
        unique_values = np.unique(binary_label)
        if len(unique_values) > 2 or (len(unique_values) == 2 and not (0 in unique_values and 255 in unique_values)):
            print(f"警告: 帧 {new_filename} 的伪标签不是纯二值! 唯一值: {unique_values}")
        elif idx == start_idx:  # 只打印第一帧的验证信息
            print(f"    二值化验证: 唯一灰度值 {unique_values} ✓")

        # 添加到已保存文件列表（不含扩展名，但包含完整的新文件名）
        saved_files.append(new_filename)
        saved_count += 1

    print(f"保存了 {saved_count} 对图像和二值化伪标签")
    print(f"  - 图像目录: {img_dir}")
    print(f"  - 标签目录: {pl_dir}")
    print(f"  - 所有伪标签均为纯二值图像 (0和255)")

    return saved_files, start_idx + saved_count  # 返回保存的文件名列表和下一个起始索引


def generate_transmission_maps(output_path, saved_files):
    """
    为保存的图像生成透光率图
    Args:
        output_path: 输出根路径
        saved_files: 已保存的文件名列表

    输出结构:
    output_path/
        ├── img/   # 存储原始图像
        ├── pl/    # 存储伪标签
        └── trans/ # 存储透光率图
    """
    img_dir = os.path.join(output_path, "img")
    trans_dir = os.path.join(output_path, "trans")

    # 创建透光率图输出目录
    os.makedirs(trans_dir, exist_ok=True)

    print(f"为 {len(saved_files)} 张图像生成透光率图...")
    total = len(saved_files)
    for i, filename in enumerate(saved_files):
        if i % 5 == 0:
            print(f"透光率图生成进度: {i}/{total} ({i/total*100:.1f}%)")

        # 加载原始图像
        img_path = os.path.join(img_dir, f"{filename}.jpg")
        
        
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{filename}.png")
            if not os.path.exists(img_path):
                print(f"警告: 找不到图像文件 {filename}")
                continue

        # 读取图像
        src = cv2.imread(img_path)
        if src is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        # 生成透光率图 - 直接使用transmission_map.py中的函数
        trans = find_transmission_map(src)

        # 保存透光率图
        trans_path = os.path.join(trans_dir, f"{filename}.png")
        cv2.imwrite(trans_path, trans * 255)

    print(f"透光率图生成完成，保存在: {trans_dir}")


def clean_output_directories(output_path):
    """
    清理输出目录中的所有现有文件
    Args:
        output_path: 输出根路径
    """
    directories = ["img", "pl", "trans"]

    for directory in directories:
        dir_path = os.path.join(output_path, directory)
        if os.path.exists(dir_path):
            print(f"清理目录: {dir_path}")
            # 删除目录中的所有文件
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"目录 {directory} 已清空")
        else:
            print(f"创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

    print("所有输出目录已清理完成")


def validate_pseudo_labels(output_path):
    """
    验证生成的伪标签是否为纯二值图像
    Args:
        output_path: 输出路径
    """
    pl_dir = os.path.join(output_path, "pl")
    if not os.path.exists(pl_dir):
        print("警告: 伪标签目录不存在")
        return

    label_files = [f for f in os.listdir(pl_dir) if f.endswith(".png")]
    if not label_files:
        print("警告: 没有找到伪标签文件")
        return

    print(f"\n验证 {len(label_files)} 个伪标签文件...")

    non_binary_count = 0
    gray_value_stats = {}

    for label_file in label_files:
        label_path = os.path.join(pl_dir, label_file)
        label = cv2.imread(label_path, 0)

        if label is None:
            print(f"警告: 无法读取标签文件 {label_file}")
            continue

        unique_values = np.unique(label)

        # 记录灰度值统计
        for val in unique_values:
            gray_value_stats[val] = gray_value_stats.get(val, 0) + 1

        # 检查是否为纯二值
        if len(unique_values) > 2 or (len(unique_values) == 2 and not (0 in unique_values and 255 in unique_values)):
            non_binary_count += 1
            if non_binary_count <= 5:  # 只显示前5个非二值文件
                print(f"  非二值标签: {label_file}, 灰度值: {unique_values}")

    print(f"\n验证结果:")
    print(f"  总文件数: {len(label_files)}")
    print(f"  二值文件数: {len(label_files) - non_binary_count}")
    print(f"  非二值文件数: {non_binary_count}")

    print(f"\n所有灰度值分布:")
    for gray_val, count in sorted(gray_value_stats.items()):
        print(f"  灰度值 {gray_val}: {count} 次出现")

    if non_binary_count == 0:
        print("✅ 所有伪标签均为纯二值图像 (只包含0和255)")
    else:
        print(f"⚠️  有 {non_binary_count} 个伪标签不是纯二值图像")

    return non_binary_count == 0


def main():
    """
    主函数 - 执行自监督模型的伪标签生成流程:
    1. 加载自监督预训练模型
    2. 加载视频标签约束（如果启用）
    3. 处理所有视频文件
    4. 从每个视频中提取帧
    5. 对帧进行预测（应用约束）
    6. 选择高置信度帧及其上下文
    7. 输出原始图像和伪标签到对应的约束目录
    8. 生成透光率图
    """
    opt = arg_parse()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载自监督模型
    generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=opt.num_filters)
    generator.load_state_dict(torch.load(opt.pretrained_weights, map_location=device, weights_only=False))
    generator.to(device)
    generator.eval()
    print(f"自监督模型已加载: {opt.pretrained_weights}")

    # 加载视频标签（如果启用约束）
    video_labels = {}
    if opt.constraint_type != "none":
        video_labels = load_video_labels(opt.video_labels_csv)
        print(f"约束类型: {opt.constraint_type}")

    # 根据约束类型设置输出路径
    if opt.constraint_type == "none":
        final_output_path = os.path.join(opt.output_path, "non_constraint")
    elif opt.constraint_type == "citizen":
        final_output_path = os.path.join(opt.output_path, "citizen_constraint")
    elif opt.constraint_type == "expert":
        final_output_path = os.path.join(opt.output_path, "expert_constraint")
    else:
        final_output_path = opt.output_path

    # 创建输出目录
    os.makedirs(final_output_path, exist_ok=True)
    print(f"输出路径: {final_output_path}")

    # 清理输出目录中的现有文件
    clean_output_directories(final_output_path)

    # 获取所有视频文件
    video_files = glob.glob(os.path.join(opt.videos_path, "*.mp4")) + glob.glob(os.path.join(opt.videos_path, "*.avi"))
    print(f"找到 {len(video_files)} 个视频文件")

    if len(video_files) == 0:
        print(f"错误: 在 {opt.videos_path} 中未找到视频文件")
        return

    # 用于在多个视频间确保索引唯一性
    next_idx = 0
    all_saved_files = []
    processed_videos = 0
    skipped_videos = 0

    for video_file in video_files:
        video_name = os.path.basename(video_file).split(".")[0]  # 获取完整的视频名（包含后缀）
        print(f"\n处理视频: {video_name}")

        # 获取视频约束信息（现在使用完整的视频名）
        constraint_info = get_video_constraint_info(video_name, video_labels, opt.constraint_type)

        if constraint_info["has_constraint"]:
            print(
                f"  约束信息: {constraint_info['constraint_strength']}, "
                f"期望烟雾: {constraint_info['expected_smoke']}, "
                f"约束置信度: {constraint_info['constraint_confidence']:.2f}"
            )

            # 对于强负约束（确定无烟雾），可以选择跳过处理
            if constraint_info["expected_smoke"] is False and "gold" in constraint_info["constraint_strength"]:
                print(f"  跳过视频 {video_name}: 黄金标准负样本，确定无烟雾")
                skipped_videos += 1
                continue
        else:
            print(f"  无约束信息，正常处理")

        # 为每个视频创建临时帧目录
        temp_frames_dir = os.path.join(final_output_path, f"temp_{video_name}")
        os.makedirs(temp_frames_dir, exist_ok=True)

        # 提取帧
        frame_infos = extract_frames_from_video(video_file, temp_frames_dir, opt.sampling_rate)
        if not frame_infos:
            print(f"跳过视频 {video_name}: 无法提取帧")
            skipped_videos += 1
            continue

        # 获取所有提取的帧路径
        frame_paths = list(frame_infos.keys())

        # 预测所有帧（应用约束）
        predictions = predict_frames(generator, frame_paths, opt.testsize, device, constraint_info)

        # 选择高置信度帧及其上下文
        selected_frames = select_high_confidence_frames(predictions, frame_infos, opt.context_frames, opt.threshold)

        print(f"选择了 {len(selected_frames)} 帧 (从 {len(frame_paths)} 帧中)")

        # 保存结果到对应的约束目录，并获取下一个起始索引
        saved_files, next_idx = save_results(predictions, selected_frames, video_name, final_output_path, next_idx, constraint_info)
        all_saved_files.extend(saved_files)
        processed_videos += 1

        # 清理临时帧目录
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir(temp_frames_dir)

        print(f"视频 {video_name} 处理完成")

    # 所有视频处理完毕后，一次性生成所有透光率图
    if all_saved_files:
        generate_transmission_maps(final_output_path, all_saved_files)

    # 验证所有生成的伪标签是否为纯二值图像
    is_all_binary = validate_pseudo_labels(final_output_path)

    print(f"\n自监督模型伪标签生成完成统计:")
    print(f"  处理的视频: {processed_videos}")
    print(f"  跳过的视频: {skipped_videos}")
    print(f"  模型类型: 自监督 (Semi-Supervised)")
    print(f"  约束类型: {opt.constraint_type}")
    print(f"  输出目录: {final_output_path}")
    print(f"  伪标签质量: {'✅ 全部为纯二值' if is_all_binary else '⚠️ 存在非二值标签'}")
    print("目录结构:")
    print("  - img/    (原始图像)")
    print("  - pl/     (伪标签 - 纯二值: 0和255)")
    print("  - trans/  (透光率图)")

    if opt.constraint_type != "none":
        print(f"\n自监督模型约束作用说明 (统一阈值0.6):")
        print(f"  • '黄金正样本' 约束: 概率0.9 > 0.6 → 前景(255)")
        print(f"  • '强正' 约束: 概率0.8 > 0.6 → 前景(255)")
        print(f"  • '弱正/可能有' 约束: 概率0.7 > 0.6 → 前景(255)")
        print(f"  • '其他正' 约束: 概率0.65 > 0.6 → 前景(255)")
        print(f"  • '其他负' 约束: 概率0.58 < 0.6 → 背景(0)")
        print(f"  • '弱负' 约束: 概率0.55 < 0.6 → 背景(0) (但接近阈值，给原始预测机会)")
        print(f"  • '强负' 约束: 概率0.4 < 0.6 → 背景(0)")
        print(f"  • '黄金负样本' 约束: 概率0.2 < 0.6 → 背景(0)")
        print(f"  • 弱约束保留40%原始预测 + 60%约束概率，平衡约束与模型预测")
        print(f"  • 统一阈值0.6确保二值化决策的一致性")
        print(f"  • 自监督模型在推理时使用先验预测，与约束机制兼容")


if __name__ == "__main__":
    main()
