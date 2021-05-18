import cv2
import numpy as np
from skimage.morphology import erosion, disk, star
from skimage.util import invert
from skimage.filters import unsharp_mask, frangi


def rgb_split(img):
    red_channel = img
    green_channel = red_channel.copy()
    red_channel[:, :, 1:3] = 0
    green_channel[:, :, 0] = 0
    green_channel[:, :, 2] = 0
    return [red_channel, green_channel]


def contrast(img, val):
    img[img[:, :] > val] = 1
    img[img[:, :] < val] = 0
    return img


def create_mask(img):
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = normalize(mask, 0.28, 98)
    mask[mask[:, :] > 0.0] = 1
    mask = erosion(mask, star(9))
    mask = erosion(mask, disk(9))
    return invert(mask)


def put_mask(img, mask):
    img[mask > 0] = 0
    return img


def normalize(img, x, y):
    min_p = np.percentile(img, x)
    max_p = np.percentile(img, y)
    img = (img - min_p) / (max_p - min_p)
    img[img[:, :] > 1] = 1
    img[img[:, :] < 0] = 0
    return img


def correcting(img):
    img = erosion(img, disk(1))
    return img


def clear_data(img):
    img = unsharp_mask(img, 25, 3)
    img = img ** 2
    img = invert(img)
    img = contrast(img, 0.95)
    return img


def find_veils(img):
    img = frangi(img, sigmas=range(1, 10, 2), alpha=2.0)
    return img
