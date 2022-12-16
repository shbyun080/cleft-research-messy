from config import config as cfg
import csv
from PIL import Image
import numpy as np

from utils import transform_pixel


def get_cleft_data():
    images = []
    labels = []

    with open(cfg.DATA.LABEL_FILE) as f:
        csv_reader = csv.reader(f, delimiter=',')
        count = -1
        for row in csv_reader:
            count += 1
            if count == 0:
                # print(f'{row[1]}, {", ".join(row[6:])}')
                continue
            try:
                img = Image.open(cfg.DATA.IMAGE_DIR+row[1])
            except:
                print(f"Couldn't load {cfg.DATA.IMAGE_DIR+row[1]}")
                continue
            x, y = img.size
            img = img.resize((256, 256))
            try:
                label = np.asarray(row[6:], dtype='float32')
            except:
                print(f"{cfg.DATA.IMAGE_DIR+row[1]} contains NULL")
                continue
            label = label.reshape((-1, 2))
            label[:, 0] = label[:, 0]*(256/x)
            label[:, 1] = label[:, 1]*(256/y)
            images.append(img)
            labels.append(label)

    return images, labels


def get_cleft_target(im_shape=(256,256)):
    target = np.asarray(cfg.DATA.TARGET)
    for i in range(len(target)):
        target[i][0] = target[i][0]/256*im_shape[0]
        target[i][1] = target[i][1]/256*im_shape[1]
    return target


def transform_labels(label, center, scale):
    output = transform_pixel(label, center, scale, (256, 256))
    return output


if __name__ == '__main__':
    imgs, labels = get_cleft_data()
