import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
import cv2
import metrics

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--langevin_step_num_des", type=int, default=10, help="number of langevin steps for ebm")
parser.add_argument("-langevin_step_size_des", type=float, default=0.026, help="step size of EBM langevin")
parser.add_argument("--energy_form", default="identity", help="tanh | sigmoid | identity | softplus")
parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")
parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
parser.add_argument("--dataset_path", type=str, default="./data/ijmond_data/test/img", help="path of the folder where the test images are stored")
parser.add_argument(
    "--gt_path", type=str, default="./data/ijmond_data/test/gt", help="path of the folder where the test ground truth masks are stored"
)
parser.add_argument("--save_path", type=str, default="./results/opa2", help="path to store the masks")
parser.add_argument("--model_path", type=str, default="./models/finetune/Model_50_gen.pth", help="path of the stored weights")
opt = parser.parse_args()

dataset_path = opt.dataset_path
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load(opt.model_path))

generator.cuda()
generator.eval()


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


save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

test_loader = test_dataset(dataset_path + "/", opt.testsize)
print(test_loader.size)

mse_high = 0
mse_low = 0
f_tot_high = 0
f_tot_low = 0
iou_total_high = 0
iou_total_low = 0
count1 = 0
count2 = 0

for i in range(test_loader.size):
    print(i)
    image, HH, WW, name = test_loader.load_data()
    image = image.cuda()

    generator_pred = generator.forward(image, training=False)
    res = generator_pred
    res = F.upsample(res, size=[WW, HH], mode="bilinear", align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
    cv2.imwrite(os.path.join(save_path, name), res.astype(int))

    gt_path = "{}/{}".format(opt.gt_path, name[:-4])
    gt = test_loader.load_gt(gt_path)
    gt = np.array(gt)

    # Calculate metrics for high-opacity (255) smoke
    high_opacity_mask = gt == 255
    low_opacity_mask = gt == 100
    if high_opacity_mask.sum() > 0:  # Ensure there is high-opacity smoke to evaluate
        f_score_high = metrics.fscore(res, gt, mask_labels=high_opacity_mask, mask_preds=low_opacity_mask)
        mse_high += metrics.mse(res, gt, mask_labels=high_opacity_mask, mask_preds=low_opacity_mask)
        iou_total_high += metrics.calculate_iou(res, gt, mask_labels=high_opacity_mask, mask_preds=low_opacity_mask)
        f_tot_high += f_score_high
        count1 += 1
        print(f"High-opacity F-score: {f_score_high}")

    # Calculate metrics for low-opacity (100) smoke
    if low_opacity_mask.sum() > 0:  # Ensure there is low-opacity smoke to evaluate
        f_score_low = metrics.fscore(res, gt, mask_labels=low_opacity_mask, mask_preds=high_opacity_mask)
        mse_low += metrics.mse(res, gt, mask_labels=low_opacity_mask, mask_preds=high_opacity_mask)
        iou_total_low += metrics.calculate_iou(res, gt, mask_labels=low_opacity_mask, mask_preds=high_opacity_mask)
        f_tot_low += f_score_low
        count2 += 1
        print(f"Low-opacity F-score: {f_score_low}")


print()
print(f"High-opacity F-score: {f_tot_high / count1}")
print(f"High-opacity mMSE: {mse_high / count1}")
print(f"High-opacity mIOU: {iou_total_high / count1}")

print(f"Low-opacity F-score: {f_tot_low / count2}")
print(f"Low-opacity mMSE: {mse_low / count2}")
print(f"Low-opacity mIOU: {iou_total_low / count2}")
