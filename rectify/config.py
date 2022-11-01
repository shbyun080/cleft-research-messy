MODEL_PATH_CKPT = './weights/hrnet/imagenet/0.0.2'
MODEL_PATH = MODEL_PATH_CKPT+'_complete'
NUM_EPOCH = 600
BATCH_SIZE = 32
INPUT_SHAPE = (256, 256, 3)

GPU_NUM = 1

TRAIN_MODEL = 'hrnet_imagenet'
TRAIN_DATASET = 'imagenet'
TRAIN_PRINT_MODEL = True
TRAIN_PRINT_MODEL_NESTED = False

LOSS_FUNCTION = 'categorical'        # categorial, mse
OPTIMIZER = 'sgd'                    # sgd, adam

AFLW_SMALL = False
AFLW_SCORE = True

IMAGENET_VAL_SIZE = 10  # Percent
IMAGENET_SEED = 424
IMAGENET_TRAIN_DIR = '../data/imagenet/data/train'
IMAGENET_NORMALIZE_CPU = False
