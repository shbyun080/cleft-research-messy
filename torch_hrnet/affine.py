import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp, resize
from skimage import data
from PIL import Image

import load_pretrained
import heatmap
from utils import prepare_input
from detector import FaceDetector


def affine(img, source, target):
    src = source
    dst = target

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out = warp(img, tform, output_shape=(image.shape[0], image.shape[1]))
    return out, tform


def reverse_affine(source, tform):
    return tform.inverse(source)


def detection_net(img_paths, name='300w', is_file=False, get_bbox=False):
    assert name in ['300w', 'cleft'], f'Invalid model name: {name}'

    if name == '300w':
        model = load_pretrained.load_hrnet('300w')
    elif name == 'cleft':
        model = load_pretrained.load_hrnet('cleft')
    model.eval()

    outputs = []
    if get_bbox:
        bboxs = []
    detector = FaceDetector()
    for img_path in img_paths:
        if is_file:
            image = Image.open(img_path)
        else:
            image = img_path
        coords = detector.detect(image)[0]
        if get_bbox:
            bboxs.append(coords)
        img, center, scale, x, y = prepare_input(img_path, coords, is_file=is_file)
        preds = model(img).cpu()
        preds = heatmap.decode_preds(preds, [center], [scale]).numpy()
        preds[:, :, 0] = preds[:, :, 0] * (256 / x)
        preds[:, :, 1] = preds[:, :, 1] * (256 / y)
        outputs.append(preds[0])

    if get_bbox:
        return outputs, bboxs
    else:
        return outputs


def affine_transform(imgs, sources, target):
    outputs = []
    transforms = []
    tgt = np.reshape(target, [-1, 2])
    for i, img in enumerate(imgs):
        src = np.reshape(sources[i], [-1, 2])
        output, tform = affine(img, src, tgt)
        outputs.append(output)
        transforms.append(tform)
    return outputs, transforms


def inverse_transform(sources, tforms):
    outputs = []
    for i, src in enumerate(sources):
        output = reverse_affine(src, tforms[i])
        outputs.append(output)
    return outputs


def get_midpoint(p1, p2):
    return [(p1[0]+p2[0])/2]


def get_rectified_images(imgs, target, labels=None, is_file=False, get_bbox=False, get_eyes=False):
    """Apply Piecewise Affine Transformations

    Images will be transformed according to source points.

    Parameters
    ----------
    imgs : Filaname Array OR PIL.Image Array
        Image paths
    target : (N) array
        Flattened target coordinates
    labels : (N) array
        Provided points for source points
    is_file : Bool
        Whether or not images provides are files or PIL.Image

    Returns
    -------
    outputs : (N, H, W, C) array
        Rectified Images
    tforms : (N, f) array
        Transformation functions

    """
    if labels is None:
        if get_bbox:
            sources, bbox = detection_net(imgs, name='300w', is_file=is_file, get_bbox=get_bbox)  # Returns (256, 256) coordinates
        else:
            sources = detection_net(imgs, name='300w', is_file=is_file, get_bbox=get_bbox)  # Returns (256, 256) coordinates
    else:
        sources = labels

    images = []
    img_sizes = []
    for img_path in images:
        if is_file:
            image = Image.open(img_path)
        else:
            image = img_path
        image = np.asarray(image)
        img_sizes.append(image.size)
        image = resize(image, (256, 256))
        imgs.append(image)

    outputs, tforms = affine_transform(images, sources, target)

    if get_eyes:  #FIXME
        # 37,40 left + 43,46 right
        eyes = [([(i[37][0]+i[40][0])/2, (i[37][1]+i[40][1])/2], [(i[43][0]+i[46][0])/2, (i[43][1]+i[46][1])/2]) for i in sources]
        if get_bbox:
            return outputs, tforms, img_sizes, bbox, eyes
        return outputs, tforms, img_sizes, eyes
    elif get_bbox:
        return outputs, tforms, img_sizes, bbox
    return outputs, tforms, img_sizes


def predict(imgs, target, is_file=False):
    """Predict Landmarks using rectification

    Landmarks will be predicted through 2 detection and transformation layers.

    Parameters
    ----------
    imgs : Array of image file paths
        Image paths
    target : (2F) or (F, 2) array
        Target coordinates
    is_file : Bool
        Whether or not images provides are files or PIL.Image

    Returns
    -------
    outputs : (N, H, W, C) array
        Predicted Landmarks, Normalized. [0-1]

    """
    outputs, tforms, img_sizes = get_rectified_images(imgs, target, is_file=is_file)
    outputs = detection_net(imgs, 'cleft', is_file=is_file)
    outputs = inverse_transform(outputs, tforms)
    for i in range(len(outputs)):
        outputs[i][:, 0] = outputs[i, :, 0]*(img_sizes[i][0]/256)
        outputs[i][i, :, 1] = outputs[i, :, 1]*(img_sizes[i][1]/256)

    return outputs


if __name__ == "__main__":
    image = data.astronaut()
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] - 1.5 * 50
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))

    fig, ax = plt.subplots()
    ax.imshow(out)
    ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    ax.axis((0, out_cols, out_rows, 0))
    plt.show()
