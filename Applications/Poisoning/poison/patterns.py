import numpy as np
import matplotlib.pyplot as plt


def cross_pattern(img_shape, cross_size=2, cross_value=0.5, center=False, offset=0):
    """Simple backdoor pattern: cross pattern (X) in the lower right corner"""
    backdoor_pattern = np.zeros(img_shape)
    _, rows, cols, _ = img_shape
    if center:
        row_anchor = rows // 2
        col_anchor = cols // 2
    elif offset > 0:
        row_anchor = rows - offset
        col_anchor = cols - offset
    else:
        row_anchor = rows
        col_anchor = cols

    for i in range(cross_size + 1):
        # moving from bottom right to top left
        backdoor_pattern[0, row_anchor - 1 - i, col_anchor - 1 - i, :] = cross_value
        # moving from bottom left to top right
        backdoor_pattern[0, row_anchor - 1 - i, col_anchor - 1 - cross_size + i, :] = cross_value
    return backdoor_pattern


def distributed_pattern(img_shape, n_pixels=10, pixel_value=0.5, seed=42):
    """Distributed backdoor pattern: `n_pixels` random pixels get changed. """
    backdoor_pattern = np.zeros(img_shape)
    _, rows, cols, _ = img_shape
    np.random.seed(seed)
    bd_pixels = np.random.randint(low=0, high=rows, size=(n_pixels, 2))
    backdoor_pattern[0, bd_pixels[:, 0], bd_pixels[:, 1], :] = pixel_value
    return backdoor_pattern


def feature_pattern(img_shape, n_feat=10, pixel_value=1.0, seed=42):
    """Distributed feature backdoor pattern: `n_feat` random features get changed. """
    _, rows, cols, channels = img_shape
    np.random.seed(seed)
    backdoor_pattern = np.zeros(np.product(img_shape))
    bd_feat = np.random.randint(low=0, high=backdoor_pattern.shape[0], size=n_feat)
    backdoor_pattern[bd_feat] = pixel_value
    backdoor_pattern = backdoor_pattern.reshape(img_shape)
    return backdoor_pattern


def noise_pattern(img_shape, l_inf_norm=0.1, seed=42):
    """Noise backdoor pattern: generate uniform noise with bounded infinity norm. """
    np.random.seed(seed)
    _, rows, cols, channels = img_shape
    backdoor_pattern = np.random.uniform(low=0.0, high=l_inf_norm, size=(1, rows, cols, channels))
    return backdoor_pattern


def load_pattern(pattern_file):
    """ Load a backdoor pattern from an image file. """
    arr = plt.imread(pattern_file)
    if arr.shape[-1] == 4:
        # remove optional alpha channel
        arr = arr[:, :, :-1]
    return arr.reshape(1, *arr.shape)


def dump_pattern(arr, pattern_file):
    """ Save the pattern in an image file. """
    plt.imsave(pattern_file, arr)


def add_pattern(array, bd_pattern, remove=False):
    """ Add (or remove) a backdoor pattern to/from a single image. """
    array_cpy = array.copy()
    # add/remove mask
    if remove:
        array_cpy -= bd_pattern
    else:
        array_cpy += bd_pattern
    array_cpy = np.clip(array_cpy, 0, 1)
    return array_cpy
