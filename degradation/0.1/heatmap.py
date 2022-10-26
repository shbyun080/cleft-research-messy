import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
import numpy as np
import math


def generate_target(img, pt, sigma):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def to_heatmap(labels, sigma=1.5, heatmap_size=(64,64), verbose=0):
    scaled_labels = labels*heatmap_size[0]
    heatmaps = np.zeros((len(scaled_labels), len(scaled_labels[0]), heatmap_size[0], heatmap_size[1]))
    for i in range(len(scaled_labels)):
        if verbose and i%(len(scaled_labels)//20)==0:
            print(f"Heatmap Generation: {i*100/len(scaled_labels)}% Done...")
        for j in range(len(scaled_labels[0])):
            heatmaps[i, j] = generate_target(heatmaps[i, j], scaled_labels[i, j, :], sigma)
    return heatmaps

def get_preds(scores):
    """
    get predictions from score maps
    """
    maxval, idx = tf.math.reduce_max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = tf.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def decode_preds(output, res=[64, 64]):
    coords = get_preds(output)  # float type

    # pose-processing
    for n in range(coords.get_shape()[0]):
        for p in range(coords.get_shape()[1]):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = tf.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = tf.identity(coords)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds