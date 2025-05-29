import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
import smoothness
from lscloss import *
from itertools import cycle
from cont_loss import intra_inter_contrastive_loss
from PIL import Image


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--lr_gen", type=float, default=2.5e-3, help="learning rate for generator")
    parser.add_argument("--lr_des", type=float, default=2.5e-3, help="learning rate for descriptor")
    parser.add_argument("--batchsize", type=int, default=6, help="training batch size")
    parser.add_argument("--trainsize", type=int, default=352, help="training dataset size")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="decay rate of learning rate")
    parser.add_argument("--decay_epoch", type=int, default=20, help="every n epochs decay learning rate")
    parser.add_argument("--beta", type=float, default=0.5, help="beta of Adam for generator")
    parser.add_argument("--gen_reduced_channel", type=int, default=32, help="reduced channel in generator")
    parser.add_argument("--des_reduced_channel", type=int, default=64, help="reduced channel in descriptor")
    parser.add_argument("--langevin_step_num_des", type=int, default=10, help="number of langevin steps for ebm")
    parser.add_argument("--langevin_step_size_des", type=float, default=0.026, help="step size of EBM langevin")
    parser.add_argument("--energy_form", default="identity", help="tanh | sigmoid | identity | softplus")
    parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")
    parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
    parser.add_argument("--num_filters", type=int, default=16, help="channel of for the final contrastive loss specific layer")
    parser.add_argument("--sm_weight", type=float, default=0.1, help="weight for smoothness loss")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for regularization term")
    parser.add_argument("--lat_weight", type=float, default=10.0, help="weight for latent loss")
    parser.add_argument("--vae_loss_weight", type=float, default=0.4, help="weight for vae loss")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1, help="weight for contrastive loss")
    parser.add_argument("--labeled_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/train", help="dataset path")
    parser.add_argument("--unlabeled_dataset_path", type=str, default="data/ijmond_data/train", help="dataset path")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="pretrained weights. it can be none",
    )  # models/ucnet_trans3_baseline/Model_50_gen.pth
    parser.add_argument("--save_model_path", type=str, default="models/ss_no_samples_50_norm_nosceduler_again_part2", help="dataset path")
    parser.add_argument("--aug", type=bool, default=False, help="Augmentation flag")
    parser.add_argument(
        "--inter", type=bool, default=False, help="Inter pixel (differenct image) match if True, else intra pixel (same image) match."
    )
    parser.add_argument("--no_samples", type=int, default=50, help="number of pixels to consider in the contrastive loss")
    opt = parser.parse_args()
    return opt


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")  # reduce='none'
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
        current_img = current_img.transpose((1, 2, 0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path + name, new_img)


def save_tensor_as_image(tensor, path):
    # Clamp values to ensure they are within a valid range
    tensor_clamped = torch.clamp(tensor, 0, 1)
    # Scale values from 0-1 to 0-255
    tensor_scaled = (tensor_clamped * 255).byte()
    print(np.unique(tensor_scaled))
    # # Convert to numpy
    tensor_np = tensor_scaled.cpu().numpy()
    # # Handle different dimensions (C, H, W) vs (H, W)
    if tensor_np.ndim == 3:  # If tensor is in CHW format
        tensor_np = tensor_np.transpose(1, 2, 0)  # Convert CHW to HWC for image
    print(tensor_np.dtype)
    # # Convert to Image
    cv2.imwrite("path.png", tensor_np)
    # img = Image.fromarray(tensor_np)
    # # Save the image
    # img.save(path)
    # print(f'Image saved to {path}')


## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def load_data(dataset_path, opt, aug=False):
    image_root = os.path.join(dataset_path, "img/")
    gt_root = os.path.join(dataset_path, "gt/")
    trans_map_root = os.path.join(dataset_path, "trans/")

    train_loader = get_loader(image_root, gt_root, trans_map_root, batchsize=opt.batchsize, trainsize=opt.trainsize, aug=aug)
    total_step = len(train_loader)
    return train_loader, total_step


if __name__ == "__main__":
    opt = argparser()
    print("Generator Learning Rate: {}".format(opt.lr_gen))

    # Build model
    generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=opt.num_filters)
    if opt.pretrained_weights is not None:
        print(f"Load pretrained weights: {opt.pretrained_weights}")
        generator.load_state_dict(torch.load(opt.pretrained_weights))
    generator.cuda()
    generator_params = generator.parameters()
    generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=[opt.beta, 0.999])

    # Load labeled data
    train_loader, total_step = load_data(opt.labeled_dataset_path, opt)
    print(f"Labeled dataset size: {total_step}")

    # Load pseudo labeled data
    train_loader_un, total_step_un = load_data(opt.unlabeled_dataset_path, opt, aug=opt.aug)
    train_loader_un_iter = cycle(train_loader_un)  # continuously iterate over the pseudo-labeled dataset
    print(f"Unlabeled dataset size: {total_step_un}")

    # Loss functions
    scheduler = lr_scheduler.StepLR(generator_optimizer, step_size=total_step, gamma=0.1)  # step_size=20
    size_rates = [1]  # multi-scale training
    # CE = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
    # smooth_loss = smoothness.smoothness_loss(size_average=True)
    loss_lsc = LocalSaliencyCoherence().cuda()
    loss_lsc_kernels_desc_defaults = [{"weight": 0.1, "xy": 3, "trans": 0.1}]
    loss_lsc_radius = 2
    weight_lsc = 0.01

    print("Let's go!")
    for epoch in range(1, (opt.epoch + 1)):
        # scheduler.step()
        generator.train()
        loss_record = AvgMeter()
        print("Generator Learning Rate: {}".format(generator_optimizer.param_groups[0]["lr"]))
        for i, pack in enumerate(train_loader, start=1):
            # Load a batch from the pseudo-labeled loader
            pseudo_labeled_pack = next(train_loader_un_iter)
            for rate in size_rates:
                generator_optimizer.zero_grad()
                ### Load Data ######################################
                # Unpack labeled data
                images_lb, gts_lb, trans_lb = pack
                num_labeled_data = images_lb.size(0)
                # save_tensor_as_image(gts_lb[0], 'mask_0.png')
                images_lb = Variable(images_lb)
                gts_lb = Variable(gts_lb)
                trans_lb = Variable(trans_lb)
                images_lb = images_lb.cuda()
                gts_lb = gts_lb.cuda()
                trans_lb = trans_lb.cuda()
                # Unpack pseudo-labeled data
                images_un, gts_un, trans_un = pseudo_labeled_pack
                images_un = Variable(images_un)
                gts_un = Variable(gts_un)
                trans_un = Variable(trans_un)
                images_un = images_un.cuda()
                gts_un = gts_un.cuda()
                trans_un = trans_un.cuda()
                ### Concatanate the labeled and unlabeled samples #####
                images = torch.cat((images_lb, images_un), dim=0)
                gts = torch.cat((gts_lb, gts_un), dim=0)
                trans = torch.cat((trans_lb, trans_un), dim=0)
                ### Feed the network ############################
                # multi-scale training samples
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                    trans = F.upsample(trans, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                pred_post_init, pred_post_ref, pred_prior_init, pred_piror_ref, latent_loss, out_post, out_prior = generator.forward(images, gts)
                # print(pred_post_init.shape, pred_post_ref.shape, pred_prior_init.shape, pred_piror_ref.shape, out_post.shape, out_prior.shape)
                ### Calculate contrastive loss ##################
                cont_loss = intra_inter_contrastive_loss(out_post, gts, num_samples=opt.no_samples, margin=1.0, inter=opt.inter)
                # print("Contrastive loss: ", cont_loss)
                ### Continue only with the labeled data ########################
                # re-scale data for crf loss
                trans_scale = F.interpolate(trans_lb, scale_factor=0.3, mode="bilinear", align_corners=True)
                images_scale = F.interpolate(images_lb, scale_factor=0.3, mode="bilinear", align_corners=True)
                pred_prior_init_scale = F.interpolate(pred_prior_init[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
                pred_prior_ref_scale = F.interpolate(pred_post_ref[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
                pred_post_init_scale = F.interpolate(pred_post_init[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
                pred_post_ref_scale = F.interpolate(pred_post_ref[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
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
                reg_loss = l2_regularisation(generator.xy_encoder) + l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
                # smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), grays)
                reg_loss = opt.reg_weight * reg_loss
                latent_loss = latent_loss

                sal_loss = 0.5 * (
                    structure_loss(pred_post_init[:num_labeled_data], gts_lb) + structure_loss(pred_post_ref[:num_labeled_data], gts_lb)
                )
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

                gen_loss_gsnn = 0.5 * (
                    structure_loss(pred_prior_init[:num_labeled_data], gts_lb) + structure_loss(pred_post_ref[:num_labeled_data], gts_lb)
                )
                gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior

                ### Total loss ###############################################
                gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss + opt.contrastive_loss_weight * cont_loss
                # print(gen_loss, gen_loss_cvae, gen_loss_gsnn, reg_loss, cont_loss)
                gen_loss.backward()
                generator_optimizer.step()
                # scheduler.step()

                if rate == 1:
                    loss_record.update(gen_loss.data, opt.batchsize)

            if i % 10 == 0 or i == total_step:
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}".format(
                        datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()
                    )
                )
            # break
        # break

        save_path = opt.save_model_path  #'models/ucnet_trans3_baseline_extention/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch >= 0 and epoch % 1 == 0:
            torch.save(generator.state_dict(), os.path.join(save_path, "Model" + "_%d" % epoch + "_gen.pth"))
