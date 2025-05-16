import numpy as np
import cv2
import os
import math
import zipfile
import argparse


def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def atmospheric_light(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def transmission_estimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * dark_channel(im3, sz)
    return transmission


def guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def transmission_refine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = guided_filter(gray, et, r, eps)

    return t


def recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def find_transmission_map(im):
    I = im.astype('float64') / 255
    dark = dark_channel(I, 15)
    A = atmospheric_light(I, dark)
    te = transmission_estimate(I, A, 15)
    t = transmission_refine(im, te)
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        default="ijmond")  #  ijmond SMOKE5K
    parser.add_argument('--dataset_zip', type=str, default="data/SMOKE5K.zip")
    parser.add_argument('--output',
                        type=str,
                        default="data/ijmond_data/transmission_map"
                        )  # data/ijmond_data data/SMOKE5K
    parser.add_argument('--mode', type=str, default="train")

    opt = parser.parse_args()
    dataset_name = opt.dataset_name
    dataset = opt.dataset_zip
    output = opt.output
    if dataset_name == "SMOKE5K":
        if not os.path.isdir(output):
            os.makedirs(output)
            with zipfile.ZipFile(dataset, 'r') as zip_ref:
                zip_ref.extractall(output)
        output_s = os.path.join(output, "SMOKE5K", opt.mode)
    else:
        output_s = os.path.join(output, opt.mode)

    for img_path in os.listdir(os.path.join(output_s, "img")):
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            fn = os.path.join(os.path.join(output_s, "img"), img_path)
            print(fn)
            src = cv2.imread(fn)
            trans = find_transmission_map(src)
            trans_path = os.path.join(output_s, "trans")
            if not os.path.exists(trans_path):
                os.makedirs(trans_path)
            cv2.imwrite(os.path.join(trans_path, img_path), trans * 255)
