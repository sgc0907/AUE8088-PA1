
# config cnn

import os

NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 120
OPTIMIZER_PARAMS    = {
    'type': 'SGD',
    'lr': 0.2,
    'momentum': 0.9,
    'weight_decay': 1e-4
}
SCHEDULER_PARAMS    = {
    'type': 'CosineAnnealingLR',
    'T_max': NUM_EPOCHS
}
WARMUP_EPOCHS       = 5

DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]
MIXUP_ALPHA         = 0.2
CUTMIX_ALPHA        = 1.0
LABEL_SMOOTHING     = 0.1
RAND_AUG_N, RAND_AUG_M = 2, 9

MODEL_NAME          = os.environ.get("MODEL_NAME", "resnet50")

ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '16-true'

WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = 'akami40-hanyang-university'
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-cosine{OPTIMIZER_PARAMS["lr"]:.1E}'
