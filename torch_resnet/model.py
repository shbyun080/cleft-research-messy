import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import wflw, cleft, cleft_control


class CNN6(nn.Module):
    def __init__(self):
        super(CNN6, self).__init__()

        self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv_add = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.pool_add = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, (3, 3), padding='same')
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(256, 512, (3, 3), padding='same')
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(256, 512, (3, 3), padding='same')
        self.pool5 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 196)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.conv_add(x)
        x = self.activation(x)
        x = self.pool_add(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def get_cnn6(device=torch.device("cuda:0")):
    model = CNN6()
    model.to(device)
    return model


def get_cnn6_pretrained(device=torch.device("cuda:0")):
    model = CNN6()
    model.to(device)
    model.load_state_dict(torch.load(f"./weights/cnn6_best_0_1.pth"))
    for p in model.parameters():
        p.requires_grad = True
    return model


def get_model(num_pts=98, device=torch.device("cuda:0")):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2 * num_pts, bias=True)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = True

    return model


def get_pretrained_model(num_pts=98, device=torch.device("cuda:0")):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2 * num_pts, bias=True)
    model.to(device)
    model.load_state_dict(torch.load(f"./weights/resnet_best_0_4.pth"))

    for p in model.parameters():
        p.requires_grad = True

    return model


def get_pretrained_model_transfer(num_pts=21, device=torch.device("cuda:0")):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 196, bias=True)
    model.load_state_dict(torch.load(f"./weights/resnet_best_0_4.pth"))
    model.fc = nn.Linear(model.fc.in_features, 2 * num_pts, bias=True)
    model.to(device)

    for p in model.parameters():
        p.requires_grad = True

    return model


def get_pretrained_cleft(num_pts=21, device=torch.device("cuda:0"), type="rectify"):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2 * num_pts, bias=True)
    model.to(device)
    if type=='rectify':
        model.load_state_dict(torch.load(f"./weights/wing_cleft_0_4.pth"))
    else:
        model.load_state_dict(torch.load(f"./weights/cleft_control_0_1.pth"))

    for p in model.parameters():
        p.requires_grad = True

    return model


def wing_loss(preds, ground_true, weights=None, w=10, eps=2):
    t = torch.abs(preds - ground_true)
    C = w - w * np.log(1 + w / eps)
    # if weights is None:
    #     return torch.mean(torch.where(t < w, w * torch.log(1 + t / eps), t - C))
    # else:
    #     return torch.mean(torch.where(t < w, w * torch.log(1 + t / eps), t - C) * weights)
    return torch.mean(torch.where(t < w, w * torch.log(1 + t / eps), t - C))


def train_resnet(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    counter = 0
    for images, landmarks in loader:
        counter += 1
        images = images.to(device)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(train_loss)


def validate_resnet(model, loader, loss_fn, device):
    model.eval()
    val_mse_loss = []
    val_loss = []
    for images, landmarks in loader:
        images = images.to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
            mse_loss_fn = nn.MSELoss()
            mse_loss = mse_loss_fn(pred_landmarks, landmarks)
            loss = loss_fn(pred_landmarks, landmarks)
        val_mse_loss.append(mse_loss.item())
        val_loss.append(loss.item())
    return (np.mean(val_loss), np.mean(val_mse_loss))


def train_300w():
    print("Start Training")
    device = torch.device("cuda:1")
    epoch = 120000
    ver = "0_5"

    model = get_pretrained_model(device=device)
    loss_fn = wing_loss
    # optimizer = optim.SGD(model.parameters(), lr=3e-5, weight_decay=5e-4, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    train_dataset = wflw.WFLW(type='train', size=(256, 256))
    valid_dataset = wflw.WFLW(type='valid', size=(256, 256))

    print(f"Training Dataset: {len(train_dataset)}")
    print(f"Validation Dataset: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8,
                              num_workers=4, pin_memory=True,
                              shuffle=True, drop_last=True
                              )
    valid_loader = DataLoader(valid_dataset, batch_size=8,
                              num_workers=4, pin_memory=True,
                              shuffle=False, drop_last=False
                              )

    train_losses = []
    val_losses = []
    val_mse_losses = []

    best_val_mse_losses = np.inf

    for e in range(1, epoch+1):
        print(f'Epoch {e}')

        # Train
        current_train_loss = train_resnet(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(current_train_loss)

        # Validation
        current_loss, current_mse_loss = validate_resnet(model, valid_loader, loss_fn, device)
        val_losses.append(current_loss)
        val_mse_losses.append(current_mse_loss)

        print(f'Train loss:          {train_losses[-1]:.7f}')
        print(f'Validation loss:     {val_losses[-1]:.7f}')
        print(f'Validation mse loss: {val_mse_losses[-1]:.7f}')

        losses = pd.DataFrame(
            list(zip(train_losses, val_losses, val_mse_losses)),
            columns=['Train', 'Validation', 'Validation MSE']
        )
        losses.to_csv(f'log/resnet_losses_{ver}.csv', index=False)

        # Save best model
        if val_mse_losses[-1] < best_val_mse_losses:
            best_val_mse_losses = val_mse_losses[-1]
            best_epoch = epoch
            with open(f"weights/resnet_best_{ver}.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)

    print(f'Best epoch: {best_epoch}')


def train_control():
    print("Start Training")
    device = torch.device("cuda:1")
    epoch = 120000
    ver = "0_2"

    model = get_pretrained_model_transfer(device=device)
    loss_fn = wing_loss
    # optimizer = optim.SGD(model.parameters(), lr=3e-5, weight_decay=5e-4, momentum=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    train_dataset = cleft_control.Cleft(type='train', size=(256, 256))
    valid_dataset = cleft_control.Cleft(type='valid', size=(256, 256))

    print(f"Training Dataset: {len(train_dataset)}")
    print(f"Validation Dataset: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8,
                              num_workers=4, pin_memory=True,
                              shuffle=True, drop_last=True
                              )
    valid_loader = DataLoader(valid_dataset, batch_size=8,
                              num_workers=4, pin_memory=True,
                              shuffle=False, drop_last=False
                              )

    train_losses = []
    val_losses = []
    val_mse_losses = []

    best_val_mse_losses = np.inf

    for e in range(1, epoch+1):
        print(f'Epoch {e}')

        # Train
        current_train_loss = train_resnet(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(current_train_loss)

        # Validation
        current_loss, current_mse_loss = validate_resnet(model, valid_loader, loss_fn, device)
        val_losses.append(current_loss)
        val_mse_losses.append(current_mse_loss)

        print(f'Train loss:          {train_losses[-1]:.7f}')
        print(f'Validation loss:     {val_losses[-1]:.7f}')
        print(f'Validation mse loss: {val_mse_losses[-1]:.7f}')

        losses = pd.DataFrame(
            list(zip(train_losses, val_losses, val_mse_losses)),
            columns=['Train', 'Validation', 'Validation MSE']
        )
        losses.to_csv(f'log/cleft_control_losses_{ver}.csv', index=False)

        # Save best model
        if val_mse_losses[-1] < best_val_mse_losses:
            best_val_mse_losses = val_mse_losses[-1]
            best_epoch = epoch
            with open(f"weights/cleft_control_{ver}.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)

    print(f'Best epoch: {best_epoch}')


def train_cleft():
    print("Start Training")
    device = torch.device("cuda:1")
    epoch = 120000
    ver = "0_4"

    model = get_pretrained_model_transfer(device=device)
    loss_fn = wing_loss
    # optimizer = optim.SGD(model.parameters(), lr=3e-5, weight_decay=5e-4, momentum=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    train_dataset = cleft.Cleft(type='train', size=(256, 256))
    valid_dataset = cleft.Cleft(type='valid', size=(256, 256))

    print(f"Training Dataset: {len(train_dataset)}")
    print(f"Validation Dataset: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8,
                              num_workers=4, pin_memory=True,
                              shuffle=True, drop_last=True
                              )
    valid_loader = DataLoader(valid_dataset, batch_size=8,
                              num_workers=4, pin_memory=True,
                              shuffle=False, drop_last=False
                              )

    train_losses = []
    val_losses = []
    val_mse_losses = []

    best_val_mse_losses = np.inf

    for e in range(1, epoch+1):
        print(f'Epoch {e}')

        # Train
        current_train_loss = train_resnet(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(current_train_loss)

        # Validation
        current_loss, current_mse_loss = validate_resnet(model, valid_loader, loss_fn, device)
        val_losses.append(current_loss)
        val_mse_losses.append(current_mse_loss)

        print(f'Train loss:          {train_losses[-1]:.7f}')
        print(f'Validation loss:     {val_losses[-1]:.7f}')
        print(f'Validation mse loss: {val_mse_losses[-1]:.7f}')

        losses = pd.DataFrame(
            list(zip(train_losses, val_losses, val_mse_losses)),
            columns=['Train', 'Validation', 'Validation MSE']
        )
        losses.to_csv(f'log/wing_cleft_losses_{ver}.csv', index=False)

        # Save best model
        if val_mse_losses[-1] < best_val_mse_losses:
            best_val_mse_losses = val_mse_losses[-1]
            best_epoch = epoch
            with open(f"weights/wing_cleft_{ver}.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)

    print(f'Best epoch: {best_epoch}')


if __name__ == '__main__':
    train_control()
