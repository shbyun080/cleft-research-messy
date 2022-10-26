import numpy as np

import affine


def initial_detection_net(imgs):
    # TODO run imgs through resnet to get target coords
    outputs = imgs
    return outputs      # (N, 2F) array


def final_detection_net(imgs):
    # TODO run imgs through hrnet to get final coords
    outputs = imgs
    return outputs      # (N, F, 2) array


def affine_transform(imgs, sources, target):
    outputs = []
    transforms = []
    tgt = np.reshape(target, [-1, 2])
    for i, image in enumerate(imgs):
        src = np.reshape(sources[i], [-1, 2])
        output, tform = affine.affine(image, src, tgt)
        outputs.append(output)
        transforms.append(tform)
    return outputs, transforms


def inverse_transform(sources, tforms):
    outputs = []
    for i, src in enumerate(sources):
        output = affine.reverse_affine(src, tforms[i])
        outputs.append(output)
    return outputs


def get_rectified_images(imgs, target, predict=True, labels=None):
    """Apply Piecewise Affine Transformations

    Images will be transformed according to source points.

    Parameters
    ----------
    imgs : (N, H, W, C) array
        Images.
    target : (N) array
        Flattened target coordinates
    predict : Bool
        Whether to predict source points or use provided points
    labels : (N) array
        Provided points for source points

    Returns
    -------
    outputs : (N, H, W, C) array
        Rectified Images.
    tforms : (N, f) array
        Transformation functions.

    """
    if predict:
        sources = initial_detection_net(imgs)
    else:
        sources = labels

    outputs, tforms = affine_transform(imgs, sources, target)
    return outputs, tforms


def predict(imgs, target):
    """Predict Landmarks using rectification

    Landmarks will be predicted through 2 detection and transformation layers.

    Parameters
    ----------
    imgs : (N, H, W, C) array
        Images.
    target : (2F) or (F, 2) array
        Target coordinates

    Returns
    -------
    outputs : (N, H, W, C) array
        Predicted Landmarks, Normalized. [0-1]

    """
    outputs, tforms = get_rectified_images(imgs, target)
    outputs = final_detection_net(outputs)
    outputs = inverse_transform(outputs, tforms)
    return outputs