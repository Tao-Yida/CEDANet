import torch
import os
from model.ResNet_models import Generator
from utils import generate_model_name
from dataloader import get_dataset_name_from_path
from dataloader import get_train_val_loaders
import numpy as np
import imageio  # 替换scipy.misc
import cv2
from torchvision import transforms

# 配置参数（可根据需要修改）
MODEL_PATH = "/home/ytao/Thesis/models/full-supervision/ijmond-limited-test/ijmond_data_test/ijmond_data_test_best_model.pth"  # 你的模型权重目录
DATASET_PATH = "data/ijmond_camera/SMOKE5K-full/citizen_constraint"  # 数据集路径
BATCH_SIZE = 2
TRAINSIZE = 352
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_prediction_init(pred):
    """
    可视化预测结果
    Args:
        pred: Predicted saliency map, size: [batch_size, channels, height, width]
    """
    # 遍历每个batch中的图像
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]  # 提取第kk个图像的预测结果
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0  # 将预测结果缩放到0-255范围
        pred_edge_kk = pred_edge_kk.astype(np.uint8)  # 转换为uint8类型
        save_path = "./temp/"
        name = "{:02d}_init.png".format(kk)
        os.makedirs(save_path, exist_ok=True)
        imageio.imwrite(save_path + name, pred_edge_kk)


def visualize_prediction_ref(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_ref.png".format(kk)
        os.makedirs(save_path, exist_ok=True)
        imageio.imwrite(save_path + name, pred_edge_kk)


def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_gt.png".format(kk)
        os.makedirs(save_path, exist_ok=True)
        imageio.imwrite(save_path + name, pred_edge_kk)


def visualize_original_img(rec_img):
    img_transform = transforms.Compose(
        [transforms.Normalize(mean=[-0.4850 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
    )
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk, :, :, :]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_img.png".format(kk)
        os.makedirs(save_path, exist_ok=True)
        current_img = current_img.transpose((1, 2, 0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path + name, new_img)


def visualize_prior_init(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_prior_init.png".format(kk)
        os.makedirs(save_path, exist_ok=True)
        imageio.imwrite(save_path + name, pred_edge_kk)


def visualize_prior_ref(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_prior_ref.png".format(kk)
        os.makedirs(save_path, exist_ok=True)
        imageio.imwrite(save_path + name, pred_edge_kk)


# 加载模型
print("Loading model...")
generator = Generator(channel=32, latent_dim=3)
# 直接加载指定权重
assert os.path.exists(MODEL_PATH), f"Model weights not found: {MODEL_PATH}"
print(f"Using weights: {MODEL_PATH}")
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.to(DEVICE)
generator.eval()

# 加载数据
image_root = os.path.join(DATASET_PATH, "img/")
gt_root = os.path.join(DATASET_PATH, "gt/")
trans_map_root = os.path.join(DATASET_PATH, "trans/")
train_loader, _ = get_train_val_loaders(
    image_root, gt_root, trans_map_root, batchsize=BATCH_SIZE, trainsize=TRAINSIZE, val_split=0.1, aug=False, freeze=True, random_seed=42
)

# 推理与可视化
with torch.no_grad():
    for i, pack in enumerate(train_loader):
        images, gts, trans = pack
        images = images.to(DEVICE)
        gts = gts.to(DEVICE)
        trans = trans.to(DEVICE)
        # 前向推理
        pred_post_init, pred_post_ref, pred_prior_init, pred_prior_ref, _ = generator(images, gts)
        # 可视化
        visualize_prediction_init(pred_post_init)
        visualize_prediction_ref(pred_post_ref)
        visualize_prior_init(pred_prior_init)
        visualize_prior_ref(pred_prior_ref)
        visualize_gt(gts)
        visualize_original_img(images)
        print(f"Batch {i} visualized.")
        # 只可视化一个batch，可根据需要去掉break
        break

print("Visualization complete. Results saved in ./temp/")
