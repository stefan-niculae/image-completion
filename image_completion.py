import random
import pickle
from functools import partial

import numpy as np
from PIL import Image

from utils import min_max_slice, coordinates, random_sample, constrain_index, slice_centered_in


def find_edge(is_empty):
    """ Empty pixels having a pixel around them that is not empty. """
    edge_mask = np.zeros(is_empty.shape, bool)

    for r, c in coordinates(is_empty, is_empty):  # only empty pixels are considered on the edge
        pixels_around_empty = [
            is_empty[r - 1, c],
            is_empty[r + 1, c],
            is_empty[r, c - 1],
            is_empty[r, c + 1]
        ]
        if not all(pixels_around_empty):
            edge_mask[r, c] = True

    return edge_mask


def compute_ssd(target_patch, candidate, is_empty, patch_len):
    """ The sum of squared differences between candidate texture and target patch to fill, for all non-empty pixels. """
    offset = 2 * patch_len
    ssd_matrix = np.zeros((candidate.shape[0] - offset, candidate.shape[1] - offset), np.uint64)
    candidate = candidate.astype(np.uint16)  # coerce into uint16 because we square uint8

    for r, c in coordinates(ssd_matrix):
        for patch_r, patch_c in coordinates(target_patch, ~is_empty):  # only for non-empty pixels
            target_pixel = target_patch[patch_r, patch_c]
            texture_pixel = candidate[r + patch_r, c + patch_c]

            ssd_matrix[r, c] += ((target_pixel - texture_pixel) ** 2).sum()  # times: .sum < np.sum < builtin sum

    return ssd_matrix


def copy_patch(destination, destination_center, source, source_center, is_fillable, patch_len):
    """ Transfer from source into destination (in place), for fillable pixels. """
    patch_size = 2 * patch_len + 1
    for r, c in coordinates(patch_size, is_fillable):  # only for fillable pixels
        destination_row = r + destination_center[0] - patch_len
        destination_col = c + destination_center[1] - patch_len

        source_row  = r + source_center[0]
        source_col  = c + source_center[1]

        destination[destination_row, destination_col] = source[source_row, source_col]


def preprocess_image(IMAGE_PATH, HOLE_PATH, SEARCH_PATH, PATCH_LEN):
    im = Image.open(IMAGE_PATH)
    im_array = np.asarray(im)

    with open(HOLE_PATH, 'rb') as f:  # saved via poly_select
        hole_mask = pickle.load(f).astype(bool)
    hole_indices = hole_mask.nonzero()
    assert ((min(hole_indices[0]) >= PATCH_LEN) and
            (max(hole_indices[0]) < im_array.shape[0] - PATCH_LEN) and
            (min(hole_indices[1]) >= PATCH_LEN) and
            (max(hole_indices[1]) < im_array.shape[1] - PATCH_LEN)), 'Hole is too close to edge of image for this patch size'

    punctured_im = im_array.copy()
    punctured_im[hole_indices] = False

    with open(SEARCH_PATH, 'rb') as f:
        searchable_mask = pickle.load(f).astype(bool)
    search_indices = searchable_mask.nonzero()
    searchable = im_array[min_max_slice(search_indices[0]), min_max_slice(search_indices[1])]
    assert ((searchable.shape[0] > 2 * PATCH_LEN + 1) and
            (searchable.shape[1] > 2 * PATCH_LEN + 1)), 'Texture image is smaller than patch size'

    return im, hole_mask, punctured_im, searchable, searchable_mask


def fill_hole(hole_mask, punctured_im, searchable, PATCH_LEN, PATCH_SELECTION_STD):
    """ Iteratively fill punctured_im using hole_mask (both in place), yielding intermediate results. """
    slice_centered_in_ = partial(slice_centered_in, length=PATCH_LEN)
    compute_ssd_ = partial(compute_ssd, patch_len=PATCH_LEN)
    copy_patch_ = partial(copy_patch, patch_len=PATCH_LEN)

    while hole_mask.any():
        on_edge_mask = find_edge(hole_mask)

        while on_edge_mask.any():
            yield punctured_im, hole_mask

            # pick a random pixel on the edge to be the center of the patch to fill
            patch_to_fill_center = random_sample(zip(*on_edge_mask.nonzero()))
            patch_to_fill = punctured_im[slice_centered_in_(patch_to_fill_center)]
            is_fillable_mask = hole_mask[slice_centered_in_(patch_to_fill_center)]

            ssd_matrix = compute_ssd_(patch_to_fill, searchable, is_fillable_mask)

            # select closest patch (with a bit of stochasticity)
            indices_of_closest = ssd_matrix.argsort(axis=None)  # on the raveled matrix
            selected_pos = constrain_index(random.gauss(0, PATCH_SELECTION_STD), len(indices_of_closest))
            match_center = np.unravel_index(indices_of_closest[selected_pos], ssd_matrix.shape)

            copy_patch_(punctured_im, patch_to_fill_center, searchable, match_center, is_fillable_mask)

            # since we just filled it, the pixels in the patch are neither empty, nor on the edge
            hole_mask[slice_centered_in_(patch_to_fill_center)] = False
            on_edge_mask[slice_centered_in_(patch_to_fill_center)] = False

    yield punctured_im, hole_mask
