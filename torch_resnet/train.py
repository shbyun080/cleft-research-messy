import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings
# warnings.filterwarnings("ignore")

from load_pretrained import get_cleft_model
from config import config as cfg
from affine import get_rectified_images
from heatmap import to_heatmap, decode_preds
from utils import interocular_nme, prepare_input, crop
from data import get_cleft_data, transform_labels, get_cleft_target


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.join(output_dir, filename), latest_path)

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'].module, os.path.join(output_dir, 'model_best.pth'))

def prepare(image, bbox, image_size=(256,256)):
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
    center_w = (bbox[0] + bbox[2]) / 2
    center_h = (bbox[1] + bbox[3]) / 2
    center = torch.Tensor([center_w, center_h])
    scale *= 1.25
    img = image
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = crop(img, center, scale, image_size, rot=0)
    img = img.astype(np.float32)
    return img


def train():
    output_dir = cfg.TRAIN.DIR

    target_coords = get_cleft_target()

    print("Fetching data...")
    images, labels = get_cleft_data()

    print("Rectifying Images...")
    # get rectified images and transforms for all
    images, tforms, img_sizes, bboxs, eyes = get_rectified_images(images, target_coords, get_bbox=True, get_eyes=True)

    print("Cleaning up...")
    labels = [labels[i] for i in range(len(labels)) if images[i] is not None]
    tforms = [tforms[i] for i in range(len(tforms)) if images[i] is not None]
    img_sizes = [img_sizes[i] for i in range(len(img_sizes)) if images[i] is not None]
    bboxs = [bboxs[i] for i in range(len(bboxs)) if images[i] is not None]
    eyes = [eyes[i] for i in range(len(eyes)) if images[i] is not None]
    images = [images[i] for i in range(len(images)) if images[i] is not None]

    print("Readying labels...")
    # transform all landmarks
    labels = [tforms[i](labels[i]) for i in range(len(labels))]

    bbox = [25, 90, 225, 205]

    # prepare images & labels with cropping
    centers = []
    scales = []
    for i in range(len(images)):
        img, center, scale, _, _ = prepare_input(images[i], bbox, is_numpy=True)
        images[i] = img
        for j in range(len(labels[i])):
            labels[i][j] = transform_labels(labels[i][j], center, scale)
        centers.append(center)
        scales.append(scale)

    # # Plot an image as a test
    # plt.imshow(np.moveaxis(images[5][0].cpu().detach().numpy(), 0, -1))
    # for x, y in labels[5]:
    #     plt.plot(x, y, 'r.')
    # plt.show()

    print("Generating heatmaps...")
    # generate heatmap
    targets = to_heatmap(np.array(labels)/256)

    # re_target = torch.Tensor(targets[5])
    # re_target = re_target.unsqueeze(0)
    # re_labels = decode_preds(re_target, [centers[5]], [scales[5]], transform_coords=False)
    # plt.imshow(np.moveaxis(images[5][0].cpu().detach().numpy(), 0, -1))
    # for x, y in re_labels[0]:
    #     print(x,y)
    #     plt.plot(4*x, 4*y, 'r.')
    # plt.show()

    print("Setting parameters...")
    # Split datasets
    tv_split = 30
    train_images, train_targets, train_labels = images[:tv_split], targets[:tv_split], labels[:tv_split]
    valid_images, valid_targets, valid_labels, valid_centers, valid_scales, valid_eyes = images[tv_split:], targets[tv_split:], labels[tv_split:], centers[tv_split:], scales[tv_split:], eyes[tv_split:]

    model = get_cleft_model(pretrained=cfg.TRAIN.PRETRAINED)
    model.train()

    criterion = nn.MSELoss(size_average=True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEPS, 0.1, -1)

    print(f"Begin Training with epoch: {cfg.TRAIN.EPOCH}")

    best_nme = None
    for epoch in range(cfg.TRAIN.EPOCH):
        lr_scheduler.step()

        running_loss = 0

        print(f"Training: {epoch+1}/{cfg.TRAIN.EPOCH}")
        # TRAINING LOOP
        model.train()
        for i in range(len(train_images)):  # train_len = ceil(len(train)/batch_size)
            end = i+4
            if i+4>=len(train_images):
                end = len(train_images)

            img_batch = []
            target_batch = []
            for j in range(i, end):
                img = train_images[j]
                img.to("cuda:0")
                img.cuda()

                tgt = train_targets[j]
                tgt = torch.Tensor(tgt)
                tgt = tgt.unsqueeze(0)

                img_batch.append(img)
                target_batch.append(tgt)

            img_batch = torch.cat(img_batch)
            target_batch = torch.cat(target_batch)

            optimizer.zero_grad()

            outputs = model(img_batch)
            target_batch = target_batch.cuda(non_blocking=True)

            loss = criterion(outputs, target_batch)
            running_loss += loss

            loss.backward()
            optimizer.step()

        print(f"loss: {running_loss/len(train_images)}")

        running_loss = 0

        print(f"Validation: {epoch+1}/{cfg.TRAIN.EPOCH}")
        # VALIDATION LOOP
        model.eval()
        nme = 0
        for i in range(len(valid_images)):
            img_batch = valid_images[i]

            target_batch = valid_targets[i]
            target_batch = torch.Tensor(target_batch)
            target_batch = target_batch.unsqueeze(0)

            outputs = model(img_batch)
            target_batch = target_batch.cuda(non_blocking=True)

            loss = criterion(outputs, target_batch)
            running_loss += loss

            preds = decode_preds(outputs, [valid_centers[i]], [valid_scales[i]])  # get change attribs from post-affine prepare_image

            # if epoch%500==0 and i == 0:
            #     f, ax = plt.subplots(1, 21)
            #     for j, score in enumerate(outputs[0].cpu().detach().numpy()):
            #         ax[j].imshow(score)
            #     plt.show()

            nme += interocular_nme(preds, [valid_labels[i]], [valid_eyes[i]])[0]  # get eyes from 300w prediction

        nme /= len(valid_images)
        print(f"nme: {nme}, loss: {running_loss/len(valid_images)}")

        if best_nme is None or best_nme > nme:
            print(f"Current validation NME of {nme} is the lowest. Saving checkpoint...")
            best_nme = nme
            torch.save(model.module.state_dict(), output_dir+'checkpoint_{}.pth'.format(epoch))
        elif epoch % 100 == 0:
            torch.save(model.module.state_dict(), output_dir+'checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(output_dir,'HR18-cleft.pth')
    print(f"Saving Model to {final_model_state_file}...")
    torch.save(model.module.state_dict(), final_model_state_file)
    print("Saved")


if __name__ == '__main__':
    train()
