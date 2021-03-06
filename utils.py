import random

from PIL import Image
import numpy as np


def min_max_slice(l: list):
    return slice(min(l), max(l) + 1)


def slice_centered_in(center: int, length: int):
    row, col = center
    return slice(row-length, row+length+1), slice(col-length, col+length+1)


def coordinates(matrix: np.array):
    n_rows, n_cols = matrix.shape[:2]
    for row in range(n_rows):
        for col in range(n_cols):
            yield row, col


def random_sample(iterable):
    return random.sample(list(iterable), 1)[0]


def constrain_index(x: float, array_len: int):
    i = round(abs(x))
    return min(i, array_len - 1)


def discard_channels(img: np.array):
    if len(img.shape) == 2:
        return img
    return img[:, :, 0]  # take the first channel


def convert_to_image(array: np.array, resize_factor: float =1):
    array = array.copy()

    maximum = array.max()
    if maximum > 255:  # heatmaps
        img = Image.fromarray(array / maximum, mode='LA')

    else:  # 0-255 images
        if array.dtype == bool:  # masks
            array = array * 255
        img = Image.fromarray(array.astype(np.uint8))

    if resize_factor == 1:
        return img
    new_width = img.width * resize_factor
    new_height = new_width * img.height / img.width
    return img.resize((int(new_width), int(new_height)), Image.ANTIALIAS)