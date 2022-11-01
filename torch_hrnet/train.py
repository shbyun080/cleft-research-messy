import torch
import torch.nn as nn
import torch.optim as optim
import os

from transfer_model import get_cleft_model
from config import config as cfg
from affine import get_rectified_images
from heatmap import to_heatmap, decode_preds
from utils import interocular_nme, prepare_input
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


def train():
    output_dir = cfg.TRAIN.DIR

    target_coords = get_cleft_target()

    images, labels = get_cleft_data()

    # images = []
    # labels = []  # 256,256 coordinates

    # get rectified images and transforms for all
    images, tforms, img_sizes, bboxs, eyes = get_rectified_images(images, target_coords, get_bbox=True, get_eyes=True)

    # transform all landmarks
    labels = [tforms(i) for i in labels]
    for i in range(len(bboxs)):
        bbox = bboxs[i]
        bbox = tforms([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
        bboxs[i] = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]

    # prepare images & labels with cropping
    centers = []
    scales = []
    for i in range(len(images)):
        img, center, scale, _, _ = prepare_input(images[i], bboxs[i])
        images[i] = img
        labels[i] = transform_labels(labels[i], center, scale)
        centers.append(center)
        scales.append(scale)

    # generate heatmap
    targets = to_heatmap(labels)

    tv_split = 50
    train_images, train_targets, train_labels = images[:tv_split], targets[:tv_split], labels[:tv_split]
    valid_images, valid_targets, valid_labels, valid_centers, valid_scales, valid_eyes = images[tv_split:], targets[tv_split:], labels[tv_split:], centers[tv_split:], scales[tv_split:], eyes[tv_split:]

    model = get_cleft_model()
    model.train()

    criterion = nn.MSELoss(size_average=True)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50], 0.1, -1)

    best_nme = None

    for epoch in range(cfg.TRAIN.EPOCH):
        lr_scheduler.step()

        # TRAINING LOOP
        model.train()

        for i in range(len(train_images)):  # train_len = ceil(len(train)/batch_size)
            # run image batches through model
            img_batch = train_images[i]
            target_batch = train_targets[i]
            outputs = model(img_batch)
            target_batch = target_batch.cuda(non_blocking=True)

            loss = criterion(outputs, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATION LOOP
        model.eval()

        nme = 0
        for i in range(len(valid_images)):
            img_batch = valid_images[i]
            target_batch = valid_targets[i]
            outputs = model(img_batch)
            target_batch = target_batch.cuda(non_blocking=True)

            loss = criterion(outputs, target_batch)

            preds = decode_preds(outputs, [valid_centers[i]], [valid_scales[i]])  # get change attribs from post-affine prepare_image

            nme += interocular_nme(preds, [valid_labels[i]], [valid_eyes[i]])  # get eyes from 300w prediction

        nme /= len(valid_images)

        if best_nme is None or best_nme>nme:
            best_nme = nme
            torch.save(model.module.state_dict(), os.path.join(output_dir, 'checkpoint_{}.pth'.format(epoch)))

    final_model_state_file = os.path.join(output_dir,'HR18-cleft.pth')
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    train()
