import torch.nn as nn


def change_last_layer(model, num_features):
    for param in model.module.parameters():
        param.requires_grad = False
    head = nn.Sequential(
        nn.Conv2d(
            in_channels=270,
            out_channels=270,
            kernel_size=1,
            stride=1,
            padding=0),
        nn.BatchNorm2d(270, momentum=0.01),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=270,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0)
    )

    for m in head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    model.module.head = head
    return model

