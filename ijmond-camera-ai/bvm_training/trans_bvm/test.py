import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--langevin_step_num_des", type=int, default=10, help="number of langevin steps for ebm")
parser.add_argument("-langevin_step_size_des", type=float, default=0.026, help="step size of EBM langevin")
parser.add_argument("--energy_form", default="identity", help="tanh | sigmoid | identity | softplus")
parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")
parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
parser.add_argument("--model_path", type=str, required=True, help="path to model file", default="models/ucnet_trans3/Model_50_gen.pth")
parser.add_argument(
    "--method",
    type=str,
    required=True,
    default="supervised",
    choices=["supervised", "semi_supervised", "domain_adaptation"],
    help="training method: supervised | semi_supervised | domain_adaptation",
)
parser.add_argument("--test_dataset", type=str, required=True, choices=["ijmond", "smoke5k"], help="test dataset: ijmond | smoke5k")
opt = parser.parse_args()

# 根据测试数据集设置数据路径
if opt.test_dataset == "ijmond":
    dataset_path = "data/ijmond_data/test/img/"
elif opt.test_dataset == "smoke5k":
    dataset_path = "data/SMOKE5K_Dataset/SMOKE5K/test/img/"

# 检测设备并设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)

# 鲁棒的模型加载
print("Loading model...")
try:
    if device.type == "cuda":
        state_dict = torch.load(opt.model_path)
    else:
        state_dict = torch.load(opt.model_path, map_location="cpu")

    # 过滤掉不匹配的键
    model_dict = generator.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and state_dict[k].shape == model_dict[k].shape}

    print(f"Successfully loaded {len(filtered_dict)}/{len(state_dict)} parameters")
    if len(filtered_dict) != len(state_dict):
        print("Warning: Some parameters were not loaded due to architecture mismatch")
        missing_keys = set(state_dict.keys()) - set(filtered_dict.keys())
        print(f"Missing keys: {list(missing_keys)[:5]}...")  # 只显示前5个

    generator.load_state_dict(filtered_dict, strict=False)

except Exception as e:
    print(f"Model loading failed: {e}")
    print("Please check if the model file and Generator architecture match")
    exit(1)

generator.to(device)
generator.eval()

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
    # 从模型路径中提取模型名称，去掉models/根路径
    model_rel_path = opt.model_path.replace("models/", "") if opt.model_path.startswith("models/") else opt.model_path
    model_name = os.path.splitext(os.path.basename(model_rel_path))[0]
    model_dir = os.path.dirname(model_rel_path)

    # 构建包含模型路径的保存路径
    if model_dir and model_dir != ".":
        save_path = os.path.join("./results", opt.method, opt.test_dataset, model_dir, dataset)
    else:
        save_path = os.path.join("./results", opt.method, opt.test_dataset, model_name, dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = os.path.join(dataset_path, dataset) if dataset else dataset_path
    print(f"Loading test data from: {image_root}")

    # 检查路径是否存在
    if not os.path.exists(image_root):
        print(f"Warning: Path {image_root} does not exist, skipping...")
        continue

    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(f"Processing image {i+1}/{test_loader.size}")
        image, HH, WW, name = test_loader.load_data()
        image = image.to(device)  # 使用设备无关的方法
        generator_pred = generator.forward(image, training=False)
        res = generator_pred
        res = F.upsample(res, size=[WW, HH], mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 安全的文件保存
        save_file_path = os.path.join(save_path, name)
        try:
            cv2.imwrite(save_file_path, res)
            if i % 10 == 0:  # 每10张图片打印一次进度
                print(f"Saved: {save_file_path}")
        except Exception as e:
            print(f"Failed to save {save_file_path}: {e}")
