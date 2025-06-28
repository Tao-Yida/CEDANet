import cv2
import numpy as np


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def sobel_energy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(grad_mag)


def tenengrad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx**2 + gy**2
    return np.mean(fm)


def brenner_focus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum((gray[:-2, :] - gray[2:, :]) ** 2)
