import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from .data import test_dataset
import cv2
import numpy as np  # 添加numpy导入，用于计算指标


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--langevin_step_num_des", type=int, default=10, help="number of langevin steps for ebm")
parser.add_argument("-langevin_step_size_des", type=float, default=0.026, help="step size of EBM langevin")
parser.add_argument("--energy_form", default="identity", help="tanh | sigmoid | identity | softplus")
parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")
parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
opt = parser.parse_args()

# dataset_path = 'SMOKE5K/SMOKE5K/test/img/'
dataset_path = "data/SMOKE5K_Dataset/SMOKE5K/test/img/"
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=16)
generator.load_state_dict(torch.load("./models/ss__no_samples_1000/Model_16_gen.pth"))

generator.cuda()
generator.eval()

# 初始化累积指标
sum_TP = 0
sum_FP = 0
sum_FN = 0
total_images = 0

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
    # save_path = './results/ucnet/' + dataset+'/'
    save_path = "./results/day/" + dataset + "/"
    # save_path = './results/ResNet50/holo/train/left/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset
    # 对应GT文件夹
    gt_root = dataset_path.replace("img/", "gt_/") + dataset + "/"
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(f"Processing image {i + 1}/{test_loader.size} in dataset {dataset}...")
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        generator_pred = generator.forward(image, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW, HH], mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res)
        # 读取并二值化GT
        gt_path = os.path.join(gt_root, name)
        gt_mask = cv2.imread(gt_path, 0)
        if gt_mask is None:
            continue
        gt_bin = (gt_mask > 128).astype(np.uint8)
        pred_bin = (res > 128).astype(np.uint8)
        # 计算TP, FP, FN
        TP = np.logical_and(pred_bin, gt_bin).sum()
        FP = np.logical_and(pred_bin, 1 - gt_bin).sum()
        FN = np.logical_and(1 - pred_bin, gt_bin).sum()
        sum_TP += TP
        sum_FP += FP
        sum_FN += FN
        total_images += 1

# 循环结束后，计算并输出指标
precision = sum_TP / (sum_TP + sum_FP + 1e-8)
recall = sum_TP / (sum_TP + sum_FN + 1e-8)
f1_score = 2 * precision * recall / (precision + recall + 1e-8)
miou = sum_TP / (sum_TP + sum_FP + sum_FN + 1e-8)
print(f"Processed {total_images} images. Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, mIoU: {miou:.4f}")
