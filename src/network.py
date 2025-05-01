# src/network.py
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
import torch
from torchvision import models
from torchvision.models.alexnet import AlexNet

# Custom packages
from src.metric import MyAccuracy
import src.config as cfg
from src.util import show_setting


class MyNetwork(AlexNet):
    def __init__(self, num_classes=200, dropout=0.5):
        super().__init__()
        # features, avgpool, classifier unchanged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
                 model_kwargs: Dict = None,
        ):
        super().__init__()
        model_kwargs = model_kwargs or {}

        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        else:
            if model_name == 'swin_b':
                # Swin-B: 사전학습 가중치 로드 후 head 교체
                from torchvision.models import Swin_B_Weights
                weights = Swin_B_Weights.IMAGENET1K_V1
                m = models.swin_b(weights=weights)
                # classifier head 교체
                in_features = m.head.in_features
                m.head = nn.Linear(in_features, num_classes)
                self.model = m
            elif model_name.startswith('swin_'):
                # 기타 Swin: 기본 get_model 사용
                self.model = models.get_model(
                    model_name,
                    num_classes=num_classes,
                    pretrained=True,
                    **model_kwargs
                )
            else:
                # 일반 모델: get_model with pretrained
                self.model = models.get_model(
                    model_name,
                    num_classes=num_classes,
                    pretrained=True,
                    **model_kwargs
                )

        # Label smoothing 적용한 Loss
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
        # Metric
        self.accuracy = MyAccuracy()
        # 하이퍼파라미터 저장
        self.save_hyperparameters()


    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if cfg.MIXUP_ALPHA > 0:
            # λ ~ Beta(α,α)
            lam = torch.distributions.Beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA).sample().item()
            idx = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1 - lam) * x[idx]
            y_a, y_b = y, y[idx]
            scores = self.forward(x)
            loss = lam * self.loss_fn(scores, y_a) + (1 - lam) * self.loss_fn(scores, y_b)
        else:
            loss, scores, y = self._common_step((x, y))
        accuracy = self.accuracy(scores, y if cfg.MIXUP_ALPHA == 0 else y_a)  # logging 용

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
