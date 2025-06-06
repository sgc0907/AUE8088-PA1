import os

MODEL_NAME          = "swin_b"
IMAGE_SIZE          = 224 
NUM_CLASSES         = 200
BATCH_SIZE          = 128 
VAL_EVERY_N_EPOCH   = 1
GRAD_CLIP_VAL       = 1.0

NUM_EPOCHS          = 200

OPTIMIZER_PARAMS    = {
    'type': 'AdamW',
    'lr': 5e-5,            
    'betas': (0.9, 0.999),
    'weight_decay': 0.05
}
SCHEDULER_PARAMS    = {
    'type': 'CosineAnnealingLR',
    'T_max': NUM_EPOCHS 
}
WARMUP_EPOCHS       = 10 

DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_MEAN          = [0.485, 0.456, 0.406]
IMAGE_STD           = [0.229, 0.224, 0.225]
MIXUP_ALPHA         = 0.2
CUTMIX_ALPHA        = 1.0
LABEL_SMOOTHING     = 0.1
RAND_AUG_N, RAND_AUG_M = 2, 9


ACCELERATOR         = 'gpu'
DEVICES             = [2]
PRECISION_STR       = '16-mixed'

WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = 'akami40-hanyang-university'
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-lr{OPTIMIZER_PARAMS["lr"]:.1E}-wd{OPTIMIZER_PARAMS["weight_decay"]:.1E}-ep{NUM_EPOCHS}' # 로그 이름 구체화