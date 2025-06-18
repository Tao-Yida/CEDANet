import torch
import torch.nn.functional as F
import os, argparse
from model.ResNet_models import Generator
import cv2
import torchvision.transforms as transforms
from PIL import Image

def argparse_fun():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--langevin_step_num_des', type=int, default=10, help='number of langevin steps for ebm')
    parser.add_argument('-langevin_step_size_des', type=float, default=0.026,help='step size of EBM langevin')
    parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
    parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
    parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
    parser.add_argument('--img_path', type=str, default="./data/ijmond_data/train/img/_1jFnujWn50-0-5-1_crop.png")
    parser.add_argument('--output_mask', type=str, default="./masks")
    parser.add_argument('--pretrained_weights', type=str, default="models/ucnet_trans3_baseline/Model_50_gen.pth")
    opt = parser.parse_args()
    return opt

class inference_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root]
        # self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_gt(self, name):
        if os.path.exists( name + ".jpg"):
            image = self.binary_loader(name + ".jpg")
        else:
            image = self.binary_loader(name + ".png")
        return image

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        self.index += 1
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

if __name__=="__main__":
    opt = argparse_fun()
    generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
    generator.load_state_dict(torch.load(opt.pretrained_weights))
    generator.cuda()
    generator.eval()

    if not os.path.exists(opt.output_mask):
        os.makedirs(opt.output_mask)

    test_loader = inference_dataset(opt.img_path, opt.testsize)
    for i in range(test_loader.size):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        generator_pred = generator.forward(image, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(os.path.join(opt.output_mask, name), res)
