import random
from functools import partial

import numpy as np
from PIL import Image

from utils import min_max_slice, coordinates, random_sample, constrain_index, slice_centered_in, discard_channels


def find_edge(is_empty):
    """ Empty pixels having a pixel around them that is not empty. """
    edge_mask = np.zeros(is_empty.shape, bool)

    for r, c in zip(*is_empty.nonzero()):  # only empty pixels are considered on the edge
        pixels_around_empty = [
            is_empty[r-1, c  ],  # left
            is_empty[r+1, c  ],  # right
            is_empty[r,   c-1],  # above
            is_empty[r,   c+1],  # below
        ]
        edge_mask[r, c] = not all(pixels_around_empty)  # at least one filled

    return edge_mask


def compute_ssd(target, candidate, is_empty, patch_side_len):
    """ The sum of squared differences between candidate texture and target patch to fill, for all non-empty pixels. """
    offset = 2 * patch_side_len  # can't put the target patch right on the edge because it will overflow
    ssd_matrix = np.zeros((candidate.shape[0] - offset, candidate.shape[1] - offset), np.uint64)

    target = target.astype(np.int32)  # coerce into uint16 because we square uint8
    is_valid = ~is_empty  # valid only if pixel is not empty

    for target_row, target_col in zip(*is_valid.nonzero()):  # go through every target patch pixel that is filled
        target_pixel = target[target_row, target_col]
        for r, c in coordinates(ssd_matrix):  # compute its distance from all pixels in the candidate
            texture_pixel = candidate[r + target_row, c + target_col]
            ssd_matrix[r, c] += ((target_pixel - texture_pixel) ** 2).sum()  # times: .sum < np.sum < builtin sum

    return ssd_matrix


def copy_patch(destination, destination_center, source, source_center, is_fillable, patch_side_len):
    """ Transfer from source into destination (in place), for fillable pixels. """
    for r, c in zip(*is_fillable.nonzero()):  # only for fillable pixels
        destination_row = r + destination_center[0] - patch_side_len
        destination_col = c + destination_center[1] - patch_side_len

        source_row  = r + source_center[0]
        source_col  = c + source_center[1]

        try:
            destination[destination_row, destination_col] = source[source_row, source_col]
        except IndexError:
            continue


def read_images(image_path, hole_path, search_path, patch_size, as_greyscale=False):
    # open image to fill
    im = Image.open(image_path)
    if as_greyscale:
        im = im.convert('L')
    im_array = np.asarray(im)

    # open where to fill
    hole_mask = np.asarray(Image.open(hole_path), bool)
    hole_mask = discard_channels(hole_mask)

    hole_indices = hole_mask.nonzero()
    assert ((min(hole_indices[0]) >= patch_size) and
            (max(hole_indices[0]) < im_array.shape[0] - patch_size) and
            (min(hole_indices[1]) >= patch_size) and
            (max(hole_indices[1]) < im_array.shape[1] - patch_size)), 'Hole is too close to edge of image for this patch size'

    # cut a hole (black) in the original image
    punctured_im = im_array.copy()
    punctured_im[hole_indices] = 0

    # where to look for replacements
    searchable_mask = np.asarray(Image.open(search_path), bool)
    searchable_mask = discard_channels(searchable_mask)
    searchable_mask[hole_mask] = False  # can't search where there is a hole

    search_indices = searchable_mask.nonzero()
    searchable = punctured_im[min_max_slice(search_indices[0]), min_max_slice(search_indices[1])]
    assert ((searchable.shape[0] > 2 * patch_size + 1) and
            (searchable.shape[1] > 2 * patch_size + 1)), 'Texture image is smaller than patch size'

    return im, hole_mask, punctured_im, searchable, searchable_mask


def fill_hole(hole_mask, punctured_im, searchable, patch_size, match_selection_std):
    """ Iteratively fill punctured_im using hole_mask (both in place), yielding intermediate results. """
    # set constants as default values to increase terseness
    slice_centered_in_ = partial(slice_centered_in, length=patch_size)
    compute_ssd_ = partial(compute_ssd, patch_side_len=patch_size)
    copy_patch_ = partial(copy_patch, patch_side_len=patch_size)

    while hole_mask.any():
        # pick a random pixel on the edge to be the center of the patch to fill
        on_edge_mask = find_edge(hole_mask)

        patch_to_fill_center = random_sample(zip(*on_edge_mask.nonzero()))
        patch_to_fill = punctured_im[slice_centered_in_(patch_to_fill_center)]
        is_fillable_mask = hole_mask[slice_centered_in_(patch_to_fill_center)]

        # select closest looking patch to fill with (with a bit of stochasticity)
        ssd_matrix = compute_ssd_(patch_to_fill, searchable, is_fillable_mask)

        indices_of_closest = ssd_matrix.argsort(axis=None)  # on the raveled matrix
        selected_pos = constrain_index(random.gauss(0, match_selection_std), len(indices_of_closest))
        match_center = np.unravel_index(indices_of_closest[selected_pos], ssd_matrix.shape)

        # copy the similar texture over the empty pixels
        copy_patch_(punctured_im, patch_to_fill_center, searchable, match_center, is_fillable_mask)
        hole_mask[slice_centered_in_(patch_to_fill_center)] = False

        yield punctured_im, hole_mask
