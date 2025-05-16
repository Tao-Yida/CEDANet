import torch
import os
import datetime

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据路径
IMG_DIR = "smoke-segmentation.v5i.coco-segmentation/test/"
MASK_DIR = "smoke-segmentation.v5i.coco-segmentation/masks/"

# 模型保存路径
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = "models"
CHECKPOINT_SUBDIR = "checkpoints"
CHECKPOINT_DIR_PATH = os.path.join(MODELS_DIR, CHECKPOINT_SUBDIR)

# 超参数
LEARNING_RATE = 3e-3
BATCH_SIZE = 8
NUM_EPOCHS = 50
IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_CLASSES = 3
VALIDATION_SPLIT = 0.1

# 学习率调度和早停
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.01
EARLY_STOPPING_PATIENCE = 10

# 类别名称
CLASS_NAMES = ["Background", "High Smoke", "Low Smoke"]

# 优化器和损失权重
WEIGHT_DECAY = 1e-5
COMBINED_LOSS_WEIGHT_CE = 1.0
COMBINED_LOSS_WEIGHT_DICE = 1.2

# 可视化结果保存路径
RESULTS_DIR = "train_curves"
