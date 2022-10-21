import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.transform import resize


def clean_up_cleft():
    cleft_landmark = pd.read_csv("cleft/").reset_index(drop=True).values  # csv file

    data = np.load("cleft/images.npz")
    images = data['img'].T
    images = np.reshape(images, (-1, 299, 299, 1))

    train_x, test_x, train_y, test_y = train_test_split(images, cleft_landmark, test_size=0.2)

    np.save("cleft/train_x.npy", train_x)
    np.save("cleft/train_y.npy", train_y)
    np.save("cleft/test_x.npy", test_x)
    np.save("cleft/test_y.npy", test_y)


def imgs_to_npz(order_file, fn=None):
    with open(order_file) as file:
        lines = file.readlines()
        lines = [l.rstrip() for l in lines]

        imgs_arr = []
        for file in lines:
            img = Image.open(file)
            img_arr = np.array(img)
            if fn is not None:
                img_arr = fn(img_arr)
            imgs_arr.append(img_arr)
        np.savez("cleft/images.npz", imgs_arr)


def resize(img):
    return resize(img, (299,299,3), anti_aliasing=True)


if __name__ == '__main__':
    imgs_to_npz("cleft/img_order.txt", fn=resize)
