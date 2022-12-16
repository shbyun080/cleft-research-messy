import PIL.Image as Image
from PIL import ImageOps
import numpy as np
import numpy.ma as ma
import csv
import matplotlib.pyplot as plt
import math
from skimage.transform import warp, warp_coords
from scipy.ndimage import map_coordinates

from data_load import get_cleft_target
from affine import get_rectified_images, detection_net


def cleft_cleanup():
    save_dir = '../data/cleft/train3/'

    img_dir = "../data/cleft/images/"
    labels_csv = "../data/cleft/labels/cleft_11_23_2022.csv"

    img_paths = []
    labels = []

    with open(labels_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        count = -1
        for row in csv_reader:
            count += 1
            if count == 0:
                continue
            try:
                img_path = row[1]
                label = np.asarray(row[6:], dtype='float32')
            except:
                continue
            label = label.reshape((-1, 2))
            img_paths.append(img_path)
            labels.append(label)

    err_count = 0
    for i in range(0, len(img_paths)):
        print(f"{i}/{len(img_paths)}")
        try:
            img, label = rectify_image2(img_dir+img_paths[i], labels[i])
            if img is None:
                err_count += 1
                print(f"Error: {err_count}/{i+1}\n\t{img_paths[i]}")
                continue

            np.save(save_dir+"landmarks/"+img_paths[i][:-4]+".pts.npy", label)
            img.save(save_dir+"images/"+img_paths[i])
        except:
            err_count += 1
            print(f"File does not exist. Error: {err_count}/{i+1}\n\t{img_paths[i]}")
            continue


def rectify_image(img_path, label):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)

    width, height = image.size
    image = image.resize((256, 256))

    label[:, 0] = label[:, 0]/width
    label[:, 1] = label[:, 1]/height

    images, tforms, img_sizes, bboxs, eyes = get_rectified_images([image], get_cleft_target(), get_bbox=True, get_eyes=True)
    image, tform, img_size, bbox, eye = images[0], tforms[0], img_sizes[0], bboxs[0], eyes[0]

    if image is None:
        return None, None

    image = np.uint8(image * 255)

    image = Image.fromarray(image, 'RGB')

    label = tform(label*256)/256

    bbox = [25, 90, 225, 205]

    image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    image = image.resize((256, 256))

    label[:, 0] = (label[:, 0] - (bbox[0]/256))/((bbox[2]-bbox[0])/256)
    label[:, 1] = (label[:, 1] - (bbox[1]/256))/((bbox[3]-bbox[1])/256)

    for x,y in label:
        if x<0 or x>=256 or y<0 or y>=256:
            print("weird image ", end='')
            return None, None

    return image, label

# n: 34,35,37,38,41,42,44,45,48,49,53,61,63,64,66,67,70,71,76,79,83,88,89,91,92,93,94,96,97,98,100,101,102,103,104,105,108,109,110,111,115,118,122,123,126,127,128,129
def rectify_image2(img_path, label):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)

    im_shape=(1024,1024)

    width, height = image.size
    image = image.resize(im_shape)
    orig_image = np.asarray(image)

    label[:, 0] = label[:, 0]/width
    label[:, 1] = label[:, 1]/height

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(image)
    # for x,y in label:
    #     ax[0].plot(x*1024,y*1024,'r.', markersize=2)

    target = get_cleft_target(im_shape=im_shape)
    images, tforms, img_sizes, bboxs, eyes = get_rectified_images([image], target, get_bbox=True, get_eyes=True, im_shape=im_shape)
    image, tform, img_size, bbox, eye = images[0], tforms[0], img_sizes[0], bboxs[0], eyes[0]

    if image is None:
        return None, None

    image = np.uint8(image * 255)

    image = Image.fromarray(image, 'RGB')
    image = image.resize(im_shape)

    #Create transfer map
    rx = np.asarray(range(1024))
    ry = np.asarray(range(1024))
    t_arr = np.array(np.meshgrid(rx, ry)).T.reshape(-1, 2)
    o_arr = tform.inverse(t_arr)
    transfer_map = np.zeros((1024,1024,2))
    for i in range(len(t_arr)):
        transfer_map[round(o_arr[i,0]), round(o_arr[i,1]), 0] = t_arr[i,0]
        transfer_map[round(o_arr[i,0]), round(o_arr[i,1]), 1] = t_arr[i,1]

    #fill gaps
    temp_map_x = ma.masked_array(transfer_map[:,:,0], transfer_map[:,:,0] == 0)
    temp_map_y = ma.masked_array(transfer_map[:,:,1], transfer_map[:,:,1] == 0)
    for shift in (-1,1):
        for axis in (0,1):
            x_shifted=np.roll(temp_map_x,shift=shift,axis=axis)
            idx=~x_shifted.mask * temp_map_x.mask
            temp_map_x[idx]=x_shifted[idx]

            y_shifted=np.roll(temp_map_y,shift=shift,axis=axis)
            idx=~y_shifted.mask * temp_map_y.mask
            temp_map_y[idx]=y_shifted[idx]

            transfer_map[:, :, 0], transfer_map[:, :, 1] = temp_map_x, temp_map_y

    #transform labels
    for i in range(len(label)):
        x = math.floor(label[i,0]*1024)
        y = math.floor(label[i,1]*1024)
        label[i,0], label[i,1] = transfer_map[x,y,0], transfer_map[x,y,1]
    label /= 1024

    # ax[1].imshow(image)
    # for x,y in label:
    #     ax[1].plot(x*1024,y*1024,'r.',markersize=2)
    #
    # ax[2].imshow(transfer_map[:,:,0])
    # plt.show()

    image = image.resize((256,256))
    bbox = [25, 90, 225, 205]

    image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    image = image.resize((256, 256))

    label[:, 0] = (label[:, 0] - (bbox[0]/256))/((bbox[2]-bbox[0])/256)
    label[:, 1] = (label[:, 1] - (bbox[1]/256))/((bbox[3]-bbox[1])/256)

    for x,y in label:
        if x<0 or x>=1 or y<0 or y>=1:
            print(f"weird image {x}, {y}", end='')
            return None, None

    return image, label


def cleft_cleanup_control():
    save_dir = '../data/cleft/control/'

    img_dir = "../data/cleft/images/"
    labels_csv = "../data/cleft/labels/cleft_11_23_2022.csv"

    img_paths = []
    labels = []

    with open(labels_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        count = -1
        for row in csv_reader:
            count += 1
            if count == 0:
                continue
            try:
                img_path = row[1]
                label = np.asarray(row[6:], dtype='float32')
            except:
                continue
            label = label.reshape((-1, 2))
            img_paths.append(img_path)
            labels.append(label)

    err_count = 0
    for i in range(len(img_paths)):
        try:
            img, label = control_image(img_dir+img_paths[i], labels[i])
            if img is None:
                err_count += 1
                print(f"Error: {err_count}/{i+1}\n\t{img_paths[i]}")
                continue
            np.save(save_dir+"landmarks/"+img_paths[i][:-4]+".pts.npy", label)
            img.save(save_dir+"images/"+img_paths[i])
        except:
            err_count += 1
            print(f"File does not exist. Error: {err_count}/{i+1}\n\t{img_paths[i]}")
            continue


def control_image(img_path, label):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)

    width, height = image.size
    image = image.resize((256, 256))

    label[:, 0] = label[:, 0]/width
    label[:, 1] = label[:, 1]/height

    _, bbox = detection_net([image], name='300w', is_file=False, get_bbox=True)
    bbox = bbox[0]

    if image is None:
        return None, None

    image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    image = image.resize((256, 256))

    label[:, 0] = (label[:, 0] - (bbox[0]/256))/((bbox[2]-bbox[0])/256)
    label[:, 1] = (label[:, 1] - (bbox[1]/256))/((bbox[3]-bbox[1])/256)

    for x,y in label:
        if x<0 or x>=256 or y<0 or y>=256:
            print("weird image ", end='')
            return None, None

    return image, label


def cleft_cleanup_base():
    save_dir = '../data/cleft/'

    img_dir = "../data/cleft/images/"
    labels_csv = "../data/cleft/labels/cleft_11_23_2022.csv"

    img_paths = []
    labels = []

    with open(labels_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        count = -1
        for row in csv_reader:
            count += 1
            if count == 0:
                continue
            try:
                img_path = row[1]
                label = np.asarray(row[6:], dtype='float32')
            except:
                continue
            label = label.reshape((-1, 2))
            img_paths.append(img_path)
            labels.append(label)

    err_count = 0
    for i in range(len(img_paths)):
        try:
            label = base_image(img_dir+img_paths[i], labels[i])
            np.save(save_dir+"landmarks/"+img_paths[i][:-4]+".pts.npy", label)
        except:
            err_count += 1
            print(f"File does not exist. Error: {err_count}/{i+1}\n\t{img_paths[i]}")
            continue


def base_image(img_path, label):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)

    width, height = image.size

    label[:, 0] = label[:, 0]/width
    label[:, 1] = label[:, 1]/height

    return label


if __name__ == "__main__":
    cleft_cleanup_base()
