import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc
from model.ResNet_models import Generator
from data import test_dataset
import cv2
import pickle

def get_args():
    """
    Set the arguments for the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--langevin_step_num_des', type=int, default=10, help='number of langevin steps for ebm')
    parser.add_argument('-langevin_step_size_des', type=float, default=0.026,help='step size of EBM langevin')
    parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
    parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
    parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
    parser.add_argument('--frames_folder', type=str, default="data/frames")
    parser.add_argument('--output_folder', type=str, default="data")
    parser.add_argument('--mode_features', action="store_true", help='Save the features')
    opt = parser.parse_args()
    return opt

def main():
    """
    Main function to run the inference of the BVM model.
    """
    opt = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the generator model
    generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
    # os.system('pwd')
    path_to_the_model = './trans_bvm/models/ucnet_trans3/Model_50_gen.pth'
    generator.load_state_dict(torch.load(path_to_the_model, map_location=torch.device(device)))
    if device == 'cuda':
        generator.cuda()
    generator.eval()

    frames_folder = opt.frames_folder
    for indx, vid in enumerate(os.listdir(frames_folder)):
        print(indx, vid)
        dataset_path = os.path.join(frames_folder, vid)
        if opt.mode_features:
            save_path = os.path.join(opt.output_folder, "features")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
           
        save_path_masks = os.path.join(opt.output_folder, "masks", vid)
        if not os.path.exists(save_path_masks):
            os.makedirs(save_path_masks)

        image_root = dataset_path + "/"
        test_loader = test_dataset(image_root, opt.testsize)
        data = {}
        for i in range(test_loader.size):
            image, HH, WW, name = test_loader.load_data()
            if device == 'cuda':
                image = image.cuda()
            generator_pred, output = generator.forward(image, training=False)
            
            if opt.mode_features:
                output = output.reshape(1,-1)
                data[name] = output.cpu().detach().numpy()
            
            res = generator_pred
            res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(os.path.join(save_path_masks, name), res)

        if opt.mode_features:
            with open(os.path.join(save_path, f"{vid}_output.pkl"), 'wb') as f:
                pickle.dump(data, f)
            
if __name__ == "__main__":
    main()
