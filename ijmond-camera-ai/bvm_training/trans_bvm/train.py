import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator, Descriptor
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
import smoothness
from lscloss import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=50, help="epoch number")  # 训练轮数
parser.add_argument("--lr_gen", type=float, default=2.5e-5, help="learning rate for generator")  # 生成器学习率
parser.add_argument("--lr_des", type=float, default=2.5e-5, help="learning rate for descriptor")  # 描述器学习率
parser.add_argument("--batchsize", type=int, default=7, help="training batch size")  # 批量大小
parser.add_argument("--trainsize", type=int, default=352, help="training dataset size")  # 训练数据集大小
parser.add_argument(
    "--clip", type=float, default=0.5, help="gradient clipping margin"
)  # 梯度裁剪边际，用于防止梯度爆炸
parser.add_argument(
    "--decay_rate", type=float, default=0.9, help="decay rate of learning rate"
)  # 学习率衰减率，用于调整学习率
parser.add_argument("--decay_epoch", type=int, default=20, help="every n epochs decay learning rate")  # 学习率衰减周期
parser.add_argument("--beta", type=float, default=0.5, help="beta of Adam for generator")  # Adam优化器的beta参数
parser.add_argument(
    "--gen_reduced_channel", type=int, default=32, help="reduced channel in generator"
)  # 生成器中减少的通道数
parser.add_argument(
    "--des_reduced_channel", type=int, default=64, help="reduced channel in descriptor"
)  # 描述器中减少的通道数
parser.add_argument(
    "--langevin_step_num_des", type=int, default=10, help="number of langevin steps for ebm"
)  # EBM的langevin步骤数，EBM是能量基模型，langevin步骤是指在生成过程中使用的迭代步骤数
parser.add_argument(
    "-langevin_step_size_des", type=float, default=0.026, help="step size of EBM langevin"
)  # EBM langevin的步长
parser.add_argument(
    "--energy_form", default="identity", help="tanh | sigmoid | identity | softplus"
)  # 能量函数的形式，tanh：双曲正切函数，sigmoid：S型函数，identity：恒等函数，softplus：平滑的ReLU函数
parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")  # 潜在维度，用于生成器和描述器的潜在空间
parser.add_argument(
    "--feat_channel", type=int, default=32, help="reduced channel of saliency feat"
)  # 重要性特征的通道数
parser.add_argument("--sm_weight", type=float, default=0.1, help="weight for smoothness loss")  # 平滑性损失的权重
parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for regularization term")  # 正则化项的权重
parser.add_argument("--lat_weight", type=float, default=10.0, help="weight for latent loss")  # 潜在损失的权重
parser.add_argument(
    "--vae_loss_weight", type=float, default=0.4, help="weight for vae loss"
)  # VAE损失的权重，VAE是变分自编码器，用于生成模型
parser.add_argument("--dataset_path", type=str, default="data/ijmond_data/train", help="dataset path")  # 数据集路径
parser.add_argument(
    "--pretrained_weights", type=str, default=None, help="pretrained weights. it can be none"
)  # 预训练权重路径，可以为None
parser.add_argument("--save_model_path", type=str, default="models/finetune", help="dataset path")  # 模型保存路径

# 所有超参数保存在opt中
opt = parser.parse_args()
print("\n========== Training Configuration ==========")
print("Training Epochs: {}".format(opt.epoch))
print("Learning Rates:")
print("  - Generator: {}".format(opt.lr_gen))
print("  - Descriptor: {}".format(opt.lr_des))
print("\nOptimization Settings:")
print("  - Batch Size: {}".format(opt.batchsize))
print("  - Training Size: {}".format(opt.trainsize))
print("  - Gradient Clip: {}".format(opt.clip))
print("  - Adam Beta: {}".format(opt.beta))
print("\nLearning Rate Schedule:")
print("  - Decay Rate: {}".format(opt.decay_rate))
print("  - Decay Epoch: {}".format(opt.decay_epoch))
print("\nModel Architecture:")
print("  - Generator Reduced Channel: {}".format(opt.gen_reduced_channel))
print("  - Descriptor Reduced Channel: {}".format(opt.des_reduced_channel))
print("  - Feature Channel: {}".format(opt.feat_channel))
print("  - Latent Dimension: {}".format(opt.latent_dim))
print("\nLoss Weights:")
print("  - Smoothness Weight: {}".format(opt.sm_weight))
print("  - Regularization Weight: {}".format(opt.reg_weight))
print("  - Latent Loss Weight: {}".format(opt.lat_weight))
print("  - VAE Loss Weight: {}".format(opt.vae_loss_weight))
print("\nPaths:")
print("  - Dataset Path: {}".format(opt.dataset_path))
print("  - Save Model Path: {}".format(opt.save_model_path))
print("  - Pretrained Weights: {}".format(opt.pretrained_weights))
print("\nEBM Settings:")
print("  - Langevin Steps: {}".format(opt.langevin_step_num_des))
print("  - Langevin Step Size: {}".format(opt.langevin_step_size_des))
print("  - Energy Form: {}".format(opt.energy_form))
print("==========================================\n")

# build models
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)  # 生成器模型

# 如果有预训练权重，则加载预训练权重，否则使用随机初始化
if opt.pretrained_weights is not None:
    print(f"Load pretrained weights: {opt.pretrained_weights}")
    generator.load_state_dict(torch.load(opt.pretrained_weights))

generator.cuda()  # 将生成器模型移动到GPU上
generator_params = generator.parameters()  # 获取生成器模型的参数，格式为可迭代对象
generator_optimizer = torch.optim.Adam(
    generator_params, lr=opt.lr_gen, betas=(opt.beta, 0.999)
)  # Adam优化器，betas的作用是控制一阶矩估计和二阶矩估计的衰减率

image_root = os.path.join(opt.dataset_path, "img/")  # data/ijmond_data/test/img
gt_root = os.path.join(opt.dataset_path, "gt/")  # data/ijmond_data/test/gt
trans_map_root = os.path.join(opt.dataset_path, "trans/")  # data/ijmond_data/test/trans

# 获取数据加载器
train_loader = get_loader(image_root, gt_root, trans_map_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# 计算数据集的总步数，训练集被分成多个batch进行训练
total_step = len(train_loader)
print(f"Dataset size: {total_step}")

scheduler = lr_scheduler.StepLR(generator_optimizer, step_size=20, gamma=0.1)
CROSSENTROPYLOSS = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  # multi-scale training
smooth_loss = smoothness.smoothness_loss(size_average=True)
loss_lsc = LocalSaliencyCoherence().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 0.1, "xy": 3, "trans": 0.1}]
loss_lsc_radius = 2
weight_lsc = 0.01


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def visualize_prediction_init(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_init.png".format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def visualize_prediction_ref(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_ref.png".format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_gt.png".format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def visualize_original_img(rec_img):
    img_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-0.4850 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
        ]
    )
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk, :, :, :]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_img.png".format(kk)
        current_img = current_img.transpose((1, 2, 0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path + name, new_img)


## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


print("Let's go!")
for epoch in range(1, (opt.epoch + 1)):
    scheduler.step()
    generator.train()
    loss_record = AvgMeter()
    print("Generator Learning Rate: {}".format(generator_optimizer.param_groups[0]["lr"]))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, trans = pack
            images = Variable(images)
            gts = Variable(gts)
            trans = Variable(trans)
            images = images.cuda()
            gts = gts.cuda()
            trans = trans.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                trans = F.upsample(trans, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
            pred_post_init, pred_post_ref, pred_prior_init, pred_piror_ref, latent_loss = generator.forward(images, gts)

            # re-scale data for crf loss
            trans_scale = F.interpolate(trans, scale_factor=0.3, mode="bilinear", align_corners=True)
            images_scale = F.interpolate(images, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_init_scale = F.interpolate(
                pred_prior_init, scale_factor=0.3, mode="bilinear", align_corners=True
            )
            pred_prior_ref_scale = F.interpolate(pred_post_ref, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_init_scale = F.interpolate(pred_post_init, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_ref_scale = F.interpolate(pred_post_ref, scale_factor=0.3, mode="bilinear", align_corners=True)
            sample = {"trans": trans_scale}

            loss_lsc_1 = loss_lsc(
                torch.sigmoid(pred_post_init_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_2 = loss_lsc(
                torch.sigmoid(pred_post_ref_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_post = weight_lsc * (loss_lsc_1 + loss_lsc_2)
            ## l2 regularizer the inference model
            reg_loss = (
                l2_regularisation(generator.xy_encoder)
                + l2_regularisation(generator.x_encoder)
                + l2_regularisation(generator.sal_encoder)
            )
            # smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), grays)
            reg_loss = opt.reg_weight * reg_loss
            latent_loss = latent_loss

            sal_loss = 0.5 * (structure_loss(pred_post_init, gts) + structure_loss(pred_post_ref, gts))
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            latent_loss = opt.lat_weight * anneal_reg * latent_loss

            loss_lsc_3 = loss_lsc(
                torch.sigmoid(pred_prior_init_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_4 = loss_lsc(
                torch.sigmoid(pred_prior_ref_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_prior = weight_lsc * (loss_lsc_3 + loss_lsc_4)

            gen_loss_cvae = sal_loss + latent_loss + loss_lsc_post
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            gen_loss_gsnn = 0.5 * (structure_loss(pred_prior_init, gts) + structure_loss(pred_post_ref, gts))
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior
            # total loss
            gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss
            gen_loss.backward()
            generator_optimizer.step()

            if rate == 1:
                loss_record.update(gen_loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}".format(
                    datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()
                )
            )

    # adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = opt.save_model_path  #'models/ucnet_trans3_baseline_extention/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch >= 0 and epoch % 10 == 0:
        torch.save(
            generator.state_dict(), os.path.join(save_path, "Model" + "_%d" % epoch + "_gen_no_pretrained_weights.pth")
        )
