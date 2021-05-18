import numpy as np
import copy
from skimage.util import invert
from skimage.filters import frangi
from skimage.morphology import star
from skimage.morphology import erosion
import cv2


def normalize(img):
    min_v = np.min(img)
    max_v = np.max(img)
    if max_v == min_v:
        return img
    return (img - min_v) / (max_v - min_v)


def rgb_split(img):
    red_channel = img
    green_channel = copy.copy(red_channel)
    red_channel[:, :, 1:3] = 0
    green_channel[:, :, 0] = 0
    green_channel[:, :, 2] = 0
    return [red_channel, green_channel]


def contrast(img, val):
    val = np.percentile(img, val)
    img[img[:, :] > val] = 1
    img[img[:, :] <= val] = 0
    return img


def create_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = normalize(img)
    img = contrast(img, 0.0)
    img = erosion(img, star(3))
    return invert(img)


def put_mask(img, mask):
    img[mask > 0.0] = 0
    return img


def do_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_green = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_clahe = clahe.apply(img_green)
    return img_clahe


def morph_filter(img, size):
    kernel = np.ones((size, size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def hb_filter(img):
    gauss = cv2.GaussianBlur(img, (7, 7), 0)
    unsharp_image = cv2.addWeighted(img, 2, gauss, -1, 0)
    return unsharp_image


def frangi_filter(img):
    return frangi(img, sigmas=range(1, 10, 2), alpha=2.0)


def hb_filter2(img):
    return unsharp_maskarp_mask(img, 20, 1.0)
