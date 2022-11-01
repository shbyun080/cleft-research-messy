import load_pretrained
import torch.nn as nn


def change_last_layer(model, num_features):
    for param in model.module.parameters():
        param.requires_grad = False

    last_layer = nn.Conv2d(
        in_channels=270,
        out_channels=num_features,
        kernel_size=1,
        stride=1,
        padding=0)
    model.module.head = nn.Sequential(*(list(model.module.head.children())[:-1]), last_layer)
    return model


def get_cleft_model():
    model = load_pretrained.load_hrnet('300w')
    model = change_last_layer(model, 21)
    return model
