import load_pretrained
from torchinfo import summary
import torch
import torch.nn as nn
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

import heatmap
from utils import prepare_input
from config import config as cfg
from affine import get_rectified_images


def get_layers(model):
    named_layers = dict(model.module.named_modules())
    return named_layers


def change_last_layer(model, num_features):
    last_layer = nn.Conv2d(
        in_channels=270,
        out_channels=num_features,
        kernel_size=1,
        stride=1,
        padding=0)
    model.module.head = nn.Sequential(*(list(model.module.head.children())[:-1]), last_layer)
    return model


def run_300w(imgs, center, scale):
    model = load_pretrained.load_hrnet('300w')
    model.eval()
    outputs = model(imgs).cpu()
    # f, ax = plt.subplots(1, len(outputs[0]))
    # for i, score in enumerate(outputs.detach()[0]):
    #     ax[i].imshow(score)
    # plt.show()
    outputs = heatmap.decode_preds(outputs, center, scale)
    return outputs.numpy()


def test_300w():
    print("fetching image...")
    img_path = ['../data/cleft/test_images/Abu Ghader_Karam (39).JPG',
                '../data/cleft/test_images/Al Araj_Ahmad_18_JAN_2020 (1).JPG',
                '../data/cleft/test_images/Abou Sadet_Karim_07_DEC_19 (9).JPG']

    img_path = img_path[0]

    img, center, scale, x, y = prepare_input(img_path, [300,1000,2500,3350], is_file=True)

    print("evaluating...")
    outputs = run_300w(img, [center], [scale])
    outputs[:,:,0] = outputs[:,:,0]*(256/x)
    outputs[:,:,1] = outputs[:,:,1]*(256/y)

    image = Image.open(img_path)
    image = np.asarray(image)
    image = resize(image, (256, 256))
    plt.imshow(image)
    for x,y in outputs[0, :27, :]:
        plt.plot(x, y, 'r.', markersize=1)
    plt.show()


def test_300w_transfer():
    model = load_pretrained.load_hrnet('300w')
    # model = change_last_layer(model, 21)
    summary(model, input_size=(1, 3, 256, 256))


def test_affine():
    print("fetching image...")
    img_path = ['../data/cleft/test_images/Abu Ghader_Karam (39).JPG',
                '../data/cleft/test_images/Al Araj_Ahmad_18_JAN_2020 (1).JPG',
                '../data/cleft/test_images/Abou Sadet_Karim_07_DEC_19 (9).JPG']

    img_path = img_path[1]

    image = Image.open(img_path)
    image = image.resize((256,256))

    outputs, tforms, orig_size = get_rectified_images([image], cfg.DATA.TARGET, is_file=False)

    f, ax = plt.subplots(1,2)
    ax[0].imshow(image)
    ax[1].imshow(outputs[0])
    plt.show()


if __name__ == '__main__':
    test_affine()
