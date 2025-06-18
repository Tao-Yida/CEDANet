import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
import cv2
import os
import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--langevin_step_num_des', type=int, default=10, help='number of langevin steps for ebm')
parser.add_argument('-langevin_step_size_des', type=float, default=0.026,help='step size of EBM langevin')
parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--dataset_path', type=str, default="./data/SMOKE5K/SMOKE5K/test/img", help='path of the fodler where the test images are stored')
parser.add_argument('--gt_path', type=str, default="./data/SMOKE5K/SMOKE5K/test/gt", help='path of the fodler where the test ground truth masks are stored')
parser.add_argument('--save_path', type=str, default="./results/SMOKE5K_no_samples_10", help='path to store the masks')
parser.add_argument('--dataset_path', type=str, default="./data/ijmond_data/test/img", help='path of the fodler where the test images are stored')
parser.add_argument('--gt_path', type=str, default="./data/ijmond_data/test/gt", help='path of the fodler where the test ground truth masks are stored')
parser.add_argument('--save_path', type=str, default="./results/testing2", help='path to store the masks')
parser.add_argument('--model_path', type=str, default="./models/ss_no_samples_50_norm_nosceduler_again/Model_42_gen.pth", help='path of the stored weights')
parser.add_argument('--num_filters', type=int, default=16, help='channel of for the final contrastive loss specific layer')
opt = parser.parse_args()

def compute_energy(disc_score):
    if opt.energy_form == 'tanh':
        energy = torch.tanh(-disc_score.squeeze())
    elif opt.energy_form == 'sigmoid':
        energy = F.sigmoid(-disc_score.squeeze())
    elif opt.energy_form == 'identity':
        energy = -disc_score.squeeze()
    elif opt.energy_form == 'softplus':
        energy = F.softplus(-disc_score.squeeze())
    return energy

generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim,  num_filters=opt.num_filters)

def evaluation(model_path):
    dataset_path = opt.dataset_path
    generator.load_state_dict(torch.load(model_path))

    generator.cuda()
    generator.eval()



    save_path =  opt.save_path #'./results/day/' + dataset+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_loader = test_dataset(dataset_path + "/", opt.testsize)
    print(test_loader.size)

    mse_ = 0
    f_tot = 0
    iou_total = 0
    for i in range(test_loader.size):
        # print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()

        generator_pred = generator.forward(image, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(os.path.join(save_path, name), res.astype(int))
        # print(os.path.join(save_path, name))
        
        gt_path = "{}/{}".format(opt.gt_path, name[:-4])
        gt = test_loader.load_gt(gt_path)
        gt = np.array(gt)
        gt[gt>0]=255
        # print(np.unique(gt))
        # print(np.unique(res))
        # break
        f_scroe = metrics.fscore( res, gt, th=0.5)

        # print(f_scroe)
        f_tot += f_scroe
        M = metrics.mse(res, gt, th=0.5)
        # print(M)
        mse_ += M
        iou = metrics.calculate_iou(res, gt, th=0.5)
        iou_total += iou
        # break

    print()

    print(f"Fscore: {f_tot/test_loader.size}")
    print(f"mMSE: {mse_/test_loader.size}")
    print(f"mIOU: {iou_total/test_loader.size}")


evaluation(opt.model_path)