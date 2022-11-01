from yacs.config import CfgNode as CN


_C = CN()

_C.PRETRAINED = CN()
_C.PRETRAINED.PATH_300W = './weights/HR18-300W.pth'
_C.PRETRAINED.PATH_IMAGENET = './weights/hrnetv2_w18_imagenet_pretrained.pth'
_C.PRETRAINED.PATH_CLEFT = './weights/cleft/0.1/HR18-cleft.pth'

_C.TRAIN = CN()
_C.TRAIN.EPOCH = 300
_C.TRAIN.DIR = './weights/cleft/0.1'

_C.DATA = CN()
_C.DATA.IMAGE_DIR = '../data/cleft/images'
_C.DATA.LABEL_FILE = '../data/cleft/labels/cleft_09_28_2022.csv'