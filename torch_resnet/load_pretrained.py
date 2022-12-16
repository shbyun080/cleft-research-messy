import torch
import torch.nn as nn

from config import config, config_pretrained
from hrnet_pretrained import HighResolutionNet


def load_hrnet(name='300w'):
    assert name in ['imagenet', '300w'], f'Invalid pretrained model name: {name}'

    model = None
    if name == 'imagenet':
        model = HighResolutionNet(config_pretrained)
        model = nn.DataParallel(model, device_ids=[0]).cuda()
        model.load_state_dict(torch.load(config.PRETRAINED.PATH_IMAGENET))
    elif name == '300w':
        model = HighResolutionNet(config_pretrained)
        model = nn.DataParallel(model, device_ids=[0]).cuda()
        weight = torch.load(config.PRETRAINED.PATH_300W)
        model.load_state_dict(weight)

    return model

