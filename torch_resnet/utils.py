import scipy
from PIL import Image
import numpy as np
from skimage.transform import resize
import torch
from scipy.spatial import distance
from statistics import mean


def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1]/2
        t_mat[1, 2] = -output_size[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, output_size, rot=0):
    center_new = center.clone()

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / output_size[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
                if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])
        else:
            # img = scipy.misc.imresize(img, [new_ht, new_wd])  # (0-1)-->(0-255)
            img = resize(img, [new_ht, new_wd], order=3)
            center_new[0] = center_new[0] * 1.0 / sf
            center_new[1] = center_new[1] * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform_pixel([0, 0], center_new, scale, output_size, invert=1))
    # Bottom right point
    br = np.array(transform_pixel(output_size, center_new, scale, output_size, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    new_img = resize(new_img, output_size, order=3)
    return new_img


def prepare_input(image, bbox, image_size=(256,256), is_file=False, is_numpy=False):
    """

    :param image:The path of the image to be detected
    :param bbox:The bbox of target face
    :param image_size: refers to config file
    :return:
    """
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    center_w = (bbox[0] + bbox[2]) / 2
    center_h = (bbox[1] + bbox[3]) / 2
    center = torch.Tensor([center_w, center_h])
    scale *= 1.25
    if is_file:
        img = Image.open(image)
    else:
        img = image

    if is_numpy:
        x, y, _ = img.shape
    else:
        x, y = img.size
        img = np.array(img.convert('RGB'), dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = crop(img, center, scale, image_size, rot=0)
    img = img.astype(np.float32)
    if not is_numpy:
        img = (img / 255.0 - mean) / std
    img = img.transpose([2, 0, 1])
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    return img, center, scale, x, y


def interocular_nme(preds, ground, eyes, get_individual=False):
    '''

    :param preds: Predicted landmarks
        (N, F, 2)
    :param ground: Groundtruth landmarks
        (N, F, 2)
    :param eyes: For each image, first item is left eye and second item is right eye
        (N, 2, 2) Array
    :param get_individual: Whether to report individual landmark losses
    :return:
    '''
    nme_list = []
    if get_individual:
        losses = []

    for i in range(len(ground)):
        normalization = distance.euclidean(eyes[i][0], eyes[i][1])
        loss = [distance.euclidean(preds[i][j], ground[i][j])/normalization for j in range(len(ground[i]))]
        nme_list.append(mean(loss))

        if get_individual:
            losses.append(loss)

    if get_individual:
        return nme_list, losses
    else:
        return nme_list
