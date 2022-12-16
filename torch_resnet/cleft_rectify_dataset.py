import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

from affine import get_rectified_images
from data_load import get_cleft_target, transform_labels
from utils import prepare_input

class CleftRectifyDataset(Dataset):
    def __init__(self, type, target_transform=None, size=(256, 256), split=30):
        self.img_dir = "../data/cleft/train/images/"
        self.labels_dir = "../data/cleft/train/landmarks/"

        self.size = size

        self.img_labels = os.listdir(self.img_dir)
        if type == 'train':
            self.img_labels = self.img_labels[:30]
        else:
            self.img_labels = self.img_labels[30:]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_labels[idx]
        image = Image.open(img_path)
        width, height = image.size
        image = image.resize(self.size)

        label_path = self.labels_dir + self.img_labels[idx][:-4] + '.pts.npy'
        label = np.load(label_path)
        label[:, 0] = label[:, 0]/width
        label[:, 1] = label[:, 1]/height

        print("Rectifying Images...")
        # get rectified images and transforms for all
        images, tforms, img_sizes, bboxs, eyes = get_rectified_images([image], get_cleft_target(), get_bbox=True, get_eyes=True)
        image, tform, img_size, bbox, eye = images[0], tforms[0], img_sizes[0], bboxs[0], eyes[0]

        if image is None:
            return None

        print("Readying labels...")
        # transform all landmarks
        label = tform(label)

        bbox = [25, 90, 225, 205]

        img, center, scale, _, _ = prepare_input(image, bbox, is_numpy=True)
        image = img
        for j in range(len(label)):
            label[j] = transform_labels(label[j], center, scale)

        image = self.transform(image)
        label = label.flatten()
        return image, label


if __name__ == "__main__":
    dataset = CleftRectifyDataset(type='train')
