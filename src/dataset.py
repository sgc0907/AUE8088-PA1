# Python packages
from termcolor import colored
from tqdm import tqdm
import os
import tarfile
import wget

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Custom packages
import src.config as cfg


class TinyImageNetDatasetModule(LightningDataModule):
    __DATASET_NAME__ = 'tiny-imagenet-200'

    def __init__(self, batch_size: int = cfg.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        if not os.path.exists(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__)):
            # download data
            print(colored("\nDownloading dataset...", color='green', attrs=('bold',)))
            filename = self.__DATASET_NAME__ + '.tar'
            wget.download(f'https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/{filename}')

            # extract data
            print(colored("\nExtract dataset...", color='green', attrs=('bold',)))
            with tarfile.open(name=filename) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(path=cfg.DATASET_ROOT_PATH, member=member)
            os.remove(filename)

    def train_dataloader(self):
        tf_train = transforms.Compose([
            # random aug
            transforms.RandAugment(num_ops=cfg.RAND_AUG_N, magnitude=cfg.RAND_AUG_M),

            # crop, resize, jitter
            transforms.RandomResizedCrop(cfg.IMAGE_NUM_CROPS, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),

            # transform
            transforms.RandomRotation(degrees=cfg.IMAGE_ROTATION),
            transforms.RandomHorizontalFlip(p=cfg.IMAGE_FLIP_PROB),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),

            transforms.ToTensor(),

            # normalize, regularization
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD), # Normalize is usually last
        ])

        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'train'), tf_train)
        msg = f"[Train]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'val'), tf_val)
        msg = f"[Val]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            pin_memory=True,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGE_MEAN, cfg.IMAGE_STD),
        ])
        dataset = ImageFolder(os.path.join(cfg.DATASET_ROOT_PATH, self.__DATASET_NAME__, 'test'), tf_test)
        msg = f"[Test]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            num_workers=cfg.NUM_WORKERS,
            batch_size=self.batch_size,
        )
