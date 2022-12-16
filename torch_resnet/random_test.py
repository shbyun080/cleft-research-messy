import load_pretrained
from torchinfo import summary
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from torchvision import transforms
import math
import os

import heatmap
from utils import prepare_input
from config import config as cfg
from affine import get_rectified_images, detection_net
from model import get_pretrained_model, get_cnn6, get_cnn6_pretrained, get_pretrained_cleft
from utils import interocular_nme
from data_load import get_cleft_target


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

    img, center, scale, x, y = prepare_input(img_path, [300, 1000, 2500, 3350], is_file=True)

    print("evaluating...")
    outputs = run_300w(img, [center], [scale])
    outputs[:, :, 0] = outputs[:, :, 0] * (256 / x)
    outputs[:, :, 1] = outputs[:, :, 1] * (256 / y)

    image = Image.open(img_path)
    image = np.asarray(image)
    image = resize(image, (256, 256))
    plt.imshow(image)
    for x, y in outputs[0, :27, :]:
        plt.plot(x, y, 'r.', markersize=1)
    plt.show()


def test_300w_transfer():
    model = load_pretrained.load_hrnet('300w')
    # model = change_last_layer(model, 21)
    summary(model, input_size=(1, 3, 256, 256))


def test_affine():
    print("fetching image...")
    img_name = 'Abou Sadet_Karim_07_DEC_19 (9)'
    img_name = 'Acevedo Ramos_Marta Veralize_01_28_2009_3'
    img_path = '../data/cleft/train/images/' + img_name + '.JPG'

    image = Image.open(img_path)
    image = image.resize((256, 256))

    outputs, tforms, orig_size = get_rectified_images([image], cfg.DATA.TARGET, is_file=False)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(outputs[0])
    plt.show()


def test_cnn6_structure():
    model = get_cnn6()
    summary(model, input_size=(1, 3, 128, 128))


def test_cnn6():
    model = get_cnn6_pretrained()
    model.eval()
    img_path = '../data/WFLW_train/images/0_Parade_Parade_0_3_2017.jpg'
    orig_image = Image.open(img_path)
    width, height = orig_image.size
    image = orig_image.resize((128, 128))
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    img = trans(image).to(torch.device("cuda:0"))
    pred = model(img.unsqueeze(0)).cpu().detach().numpy()[0]

    pred = np.reshape(pred, (-1, 2))
    pred[:, 0] *= width
    pred[:, 1] *= height

    plt.imshow(orig_image)
    for x, y in pred:
        plt.plot(x, y, 'r.')
    plt.show()


def test_cnn6_heatmap():
    model = get_cnn6_pretrained()
    model.eval()
    img_path = '../data/WFLW_train/images/0_Parade_Parade_0_3_2017.jpg'
    orig_image = Image.open(img_path)
    image = orig_image.resize((128, 128))
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    image = trans(image).to(torch.device("cuda:0"))

    model_weights = []
    conv_layers = []
    pool_layers = []
    model_children = list(model.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.MaxPool2d:
            pool_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    outputs = []
    names = []
    for i in range(len(conv_layers)):
        image = conv_layers[i](image)
        image = pool_layers[i](image)
        outputs.append(image)
        names.append(str(conv_layers[i]))
    print(len(outputs))
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    for i in range(len(processed)):
        plt.subplot(7, 7, (i + 1))
        plt.imshow(processed[i])
    plt.show()


def test_wing():
    model = get_pretrained_model()
    model.eval()
    img_path = '../data/WFLW_train/images/0_Parade_Parade_0_3_2017.jpg'
    orig_image = Image.open(img_path)
    width, height = orig_image.size
    image = orig_image.resize((256, 256))
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    img = trans(image).to(torch.device("cuda:0"))
    pred = model(img.unsqueeze(0)).cpu().detach().numpy()[0]

    pred = np.reshape(pred, (-1, 2))
    pred[:, 0] *= width
    pred[:, 1] *= height

    plt.imshow(orig_image)
    for x, y in pred:
        plt.plot(x, y, 'r.')
    plt.show()


def test_wing_heatmap():
    model = get_pretrained_model()
    model.eval()
    img_path = '../data/WFLW_train/images/0_Parade_Parade_0_3_2017.jpg'
    orig_image = Image.open(img_path)
    image = orig_image.resize((256, 256))
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    image = trans(image).to(torch.device("cuda:0"))

    model_weights = []
    conv_layers = []
    model_children = list(model.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    for i in range(len(processed)):
        plt.subplot(7, 7, (i + 1))
        plt.imshow(processed[i])
    plt.show()


def test_wing_cleft():
    model = get_pretrained_cleft(type='rectify')
    model.eval()
    img_name = 'Abou Sadet_Karim_07_DEC_19 (9)'
    img_name = 'Acevedo Ramos_Marta Veralize_01_28_2009_3'
    img_name = 'Akash_Riham_Rami_28_04_18 (4)'
    img_path = '../data/cleft/train3/images/' + img_name + '.JPG'
    label_path = '../data/cleft/train3/landmarks/' + img_name + '.pts.npy'
    label = np.load(label_path) * 256

    orig_image = Image.open(img_path)
    width, height = orig_image.size
    image = orig_image.resize((256, 256))
    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    img = trans(image).to(torch.device("cuda:0"))
    pred = model(img.unsqueeze(0)).cpu().detach().numpy()[0]

    pred = np.reshape(pred, (-1, 2))
    pred *= 256
    nme, losses = interocular_nme([pred], [label], [[[84, 66], [175, 71]]], get_individual=True)
    print(nme[0], losses[0])

    plt.imshow(orig_image)
    for x, y in pred:
        plt.plot(x, y, 'r.', markersize=2)
    for x, y in label:
        plt.plot(x, y, 'b.', markersize=2)
    plt.show()


def test_wing_cleft_full():
    fig, ax = plt.subplots(1,2)

    model = get_pretrained_cleft(type='rectify')
    model.eval()
    img_names = ['Abou Sadet_Karim_07_DEC_19 (9)',
                 'Acevedo Ramos_Marta Veralize_01_28_2009_3',
                 'Akash_Riham_Rami_28_04_18 (4)']
    img_name = img_names[1]
    img_path = '../data/cleft/images/' + img_name + '.JPG'
    label_path = '../data/cleft/landmarks/' + img_name + '.pts.npy'
    label = np.load(label_path)

    orig_image = Image.open(img_path)
    width, height = orig_image.size
    image = orig_image.resize((1024, 1024))

    # TODO rectify & crop image & get eyes for nme
    target = get_cleft_target(im_shape=(1024, 1024))
    images, tforms, img_sizes, bboxs, eyes = get_rectified_images([image], target, get_bbox=True, get_eyes=True, im_shape=(1024, 1024))
    image, tform, img_size, bbox, eye = images[0], tforms[0], img_sizes[0], bboxs[0], eyes[0]

    eye = [[[eye[0][0]/1024, eye[0][1]/1024], [eye[1][0]/1024, eye[1][1]/1024]]]

    if image is None:
        print("Error in rectification")
        return

    image = np.uint8(image * 255)

    image = Image.fromarray(image, 'RGB')
    image = image.resize((256, 256))

    #Create transfer map
    rx = np.asarray(range(1024))
    ry = np.asarray(range(1024))
    t_arr = np.array(np.meshgrid(rx, ry)).T.reshape(-1, 2)
    o_arr = tform.inverse(t_arr)
    transfer_map = np.zeros((1024,1024,2))
    for i in range(len(t_arr)):
        transfer_map[t_arr[i,0], t_arr[i,1], 0] = o_arr[i,0]
        transfer_map[t_arr[i,0], t_arr[i,1], 1] = o_arr[i,1]

    bbox = [25, 90, 225, 205]

    image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    image = image.resize((256, 256))

    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    img = trans(image).to(torch.device("cuda:0"))
    pred = model(img.unsqueeze(0)).cpu().detach().numpy()[0]

    pred = np.reshape(pred, (-1, 2))

    ax[0].imshow(image)
    for x,y in pred:
        ax[0].plot(x*256, y*256, 'r.', markersize=2)

    # TODO un-crop
    pred[:, 0] = pred[:, 0]*((bbox[2]-bbox[0])/256) + bbox[0]/256
    pred[:, 1] = pred[:, 1]*((bbox[3]-bbox[1])/256) + bbox[1]/256

    # TODO un-rectify preds
    pred *= 1024
    # pred = tform.inverse(pred)
    for i in range(len(pred)):
        x = math.floor(pred[i,0])
        y = math.floor(pred[i,1])
        xr = pred[i,0] - x
        yr = pred[i,1] - y
        pred[i,0] = transfer_map[x,y,0]
        pred[i,1] = transfer_map[x,y,1]
        pred[i,0] = (1-yr) * ((1-xr)*transfer_map[x,y,0] + xr*transfer_map[x+1,y,0]) + yr * ((1-xr)*transfer_map[x,y+1,0] + xr*transfer_map[x+1,y+1,0])
        pred[i,1] = (1-yr) * ((1-xr)*transfer_map[x,y,1] + xr*transfer_map[x+1,y,1]) + yr * ((1-xr)*transfer_map[x,y+1,1] + xr*transfer_map[x+1,y+1,1])
    pred /= 1024

    nme, losses = interocular_nme([pred], [label], eye, get_individual=True)
    print(nme[0], losses[0])

    ax[1].imshow(orig_image)
    for x, y in pred:
        ax[1].plot(x*width, y*height, 'r.', markersize=2)
    for x, y in label:
        ax[1].plot(x*width, y*height, 'b.', markersize=2)
    plt.show()


def test_wing_control_full():
    fig, ax = plt.subplots(1,2)

    model = get_pretrained_cleft(type='control')
    model.eval()
    img_names = ['Abou Sadet_Karim_07_DEC_19 (9)',
                 'Acevedo Ramos_Marta Veralize_01_28_2009_3',
                 'Akash_Riham_Rami_28_04_18 (4)']
    img_name = img_names[1]
    img_path = '../data/cleft/images/' + img_name + '.JPG'
    label_path = '../data/cleft/landmarks/' + img_name + '.pts.npy'
    label = np.load(label_path)

    orig_image = Image.open(img_path)
    width, height = orig_image.size
    image = orig_image.resize((256, 256))

    _, bbox, eyes = detection_net([image], name='300w', is_file=False, get_bbox=True, get_eyes=True)
    bbox, eye = bbox[0], eyes[0]

    if image is None:
        print("Error in detection")
        return

    image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    image = image.resize((256, 256))

    eye = [[[eye[0][0]/256, eye[0][1]/256], [eye[1][0]/256, eye[1][1]/256]]]

    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    img = trans(image).to(torch.device("cuda:0"))
    pred = model(img.unsqueeze(0)).cpu().detach().numpy()[0]

    pred = np.reshape(pred, (-1, 2))

    ax[0].imshow(image)
    for x,y in pred:
        ax[0].plot(x*256, y*256, 'r.', markersize=2)

    # TODO un-crop
    pred[:, 0] = pred[:, 0]*((bbox[2]-bbox[0])/256) + bbox[0]/256
    pred[:, 1] = pred[:, 1]*((bbox[3]-bbox[1])/256) + bbox[1]/256

    nme, losses = interocular_nme([pred], [label], eye, get_individual=True)
    print(nme[0], losses[0])

    ax[1].imshow(orig_image)
    for x, y in pred:
        ax[1].plot(x*width, y*height, 'r.', markersize=2)
    for x, y in label:
        ax[1].plot(x*width, y*height, 'b.', markersize=2)
    plt.show()


def test_wing_cleft_all(draw = False):
    if draw:
        fig, ax = plt.subplots(1,2)

    img_labels = os.listdir("../data/cleft/train3/images/")[83:]

    model = get_pretrained_cleft(type='rectify')
    model.eval()

    running_nme = 0
    running_losses = [0]*21

    for img_name in img_labels:
        print(img_name)
        img_name = img_name[:-4]
        img_path = '../data/cleft/images/' + img_name + '.JPG'
        label_path = '../data/cleft/landmarks/' + img_name + '.pts.npy'
        label = np.load(label_path)

        orig_image = Image.open(img_path)
        orig_image = ImageOps.exif_transpose(orig_image)
        width, height = orig_image.size
        image = orig_image.resize((1024, 1024))

        # TODO rectify & crop image & get eyes for nme
        target = get_cleft_target(im_shape=(1024, 1024))
        images, tforms, img_sizes, bboxs, eyes = get_rectified_images([image], target, get_bbox=True, get_eyes=True, im_shape=(1024, 1024))
        image, tform, img_size, bbox, eye = images[0], tforms[0], img_sizes[0], bboxs[0], eyes[0]

        eye = [[[eye[0][0]/1024, eye[0][1]/1024], [eye[1][0]/1024, eye[1][1]/1024]]]

        if image is None:
            print("Error in rectification")
            return

        image = np.uint8(image * 255)

        image = Image.fromarray(image, 'RGB')
        image = image.resize((256, 256))

        #Create transfer map
        rx = np.asarray(range(1024))
        ry = np.asarray(range(1024))
        t_arr = np.array(np.meshgrid(rx, ry)).T.reshape(-1, 2)
        o_arr = tform.inverse(t_arr)
        transfer_map = np.zeros((1024,1024,2))
        for i in range(len(t_arr)):
            transfer_map[t_arr[i,0], t_arr[i,1], 0] = o_arr[i,0]
            transfer_map[t_arr[i,0], t_arr[i,1], 1] = o_arr[i,1]

        bbox = [25, 90, 225, 205]

        image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        image = image.resize((256, 256))

        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        img = trans(image).to(torch.device("cuda:0"))
        pred = model(img.unsqueeze(0)).cpu().detach().numpy()[0]

        pred = np.reshape(pred, (-1, 2))

        if draw:
            ax[0].imshow(image)
            for x,y in pred:
                ax[0].plot(x*256, y*256, 'r.', markersize=2)

        # TODO un-crop
        pred[:, 0] = pred[:, 0]*((bbox[2]-bbox[0])/256) + bbox[0]/256
        pred[:, 1] = pred[:, 1]*((bbox[3]-bbox[1])/256) + bbox[1]/256

        # TODO un-rectify preds
        pred *= 1024
        # pred = tform.inverse(pred)
        for i in range(len(pred)):
            x = math.floor(pred[i,0])
            y = math.floor(pred[i,1])
            xr = pred[i,0] - x
            yr = pred[i,1] - y
            pred[i,0] = transfer_map[x,y,0]
            pred[i,1] = transfer_map[x,y,1]
            pred[i,0] = (1-yr) * ((1-xr)*transfer_map[x,y,0] + xr*transfer_map[x+1,y,0]) + yr * ((1-xr)*transfer_map[x,y+1,0] + xr*transfer_map[x+1,y+1,0])
            pred[i,1] = (1-yr) * ((1-xr)*transfer_map[x,y,1] + xr*transfer_map[x+1,y,1]) + yr * ((1-xr)*transfer_map[x,y+1,1] + xr*transfer_map[x+1,y+1,1])
        pred /= 1024

        nme, losses = interocular_nme([pred], [label], eye, get_individual=True)
        running_nme += nme[0]
        running_losses = [running_losses[i]+losses[0][i] for i in range(len(running_losses))]

        if draw:
            ax[1].imshow(orig_image)
            for x, y in pred:
                ax[1].plot(x*width, y*height, 'r.', markersize=2)
            for x, y in label:
                ax[1].plot(x*width, y*height, 'b.', markersize=2)
            plt.show()
    print(running_nme/len(img_labels), [i/len(img_labels) for i in running_losses])


def test_wing_cleft_control_all(draw = False):
    if draw:
        fig, ax = plt.subplots(1,2)

    img_labels = os.listdir("../data/cleft/train3/images/")[83:]

    model = get_pretrained_cleft(type='control')
    model.eval()

    running_nme = 0
    running_losses = [0]*21

    for img_name in img_labels:
        print(img_name)
        img_name = img_name[:-4]
        img_path = '../data/cleft/images/' + img_name + '.JPG'
        label_path = '../data/cleft/landmarks/' + img_name + '.pts.npy'
        label = np.load(label_path)

        orig_image = Image.open(img_path)
        orig_image = ImageOps.exif_transpose(orig_image)
        width, height = orig_image.size
        image = orig_image.resize((256, 256))

        _, bbox, eyes = detection_net([image], name='300w', is_file=False, get_bbox=True, get_eyes=True)
        bbox, eye = bbox[0], eyes[0]

        if bbox is None:
            print("Error in detection")
            continue

        image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        image = image.resize((256, 256))

        eye = [[[eye[0][0]/256, eye[0][1]/256], [eye[1][0]/256, eye[1][1]/256]]]

        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        img = trans(image).to(torch.device("cuda:0"))
        pred = model(img.unsqueeze(0)).cpu().detach().numpy()[0]

        pred = np.reshape(pred, (-1, 2))

        if draw:
            ax[0].imshow(image)
            for x,y in pred:
                ax[0].plot(x*256, y*256, 'r.', markersize=2)

        # TODO un-crop
        pred[:, 0] = pred[:, 0]*((bbox[2]-bbox[0])/256) + bbox[0]/256
        pred[:, 1] = pred[:, 1]*((bbox[3]-bbox[1])/256) + bbox[1]/256

        nme, losses = interocular_nme([pred], [label], eye, get_individual=True)
        running_nme += nme[0]
        running_losses = [running_losses[i]+losses[0][i] for i in range(len(running_losses))]

        if draw:
            ax[1].imshow(orig_image)
            for x, y in pred:
                ax[1].plot(x*width, y*height, 'r.', markersize=2)
            for x, y in label:
                ax[1].plot(x*width, y*height, 'b.', markersize=2)
            plt.show()

    print(running_nme/len(img_labels), [i/len(img_labels) for i in running_losses])


# if __name__ == '__main__':
#     test_wing_cleft_all(False)
#     test_wing_cleft_control_all(False)

if __name__ == '__main__':
    test_wing_cleft_full()
    test_wing_control_full()

