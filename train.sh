#!/usr/bin/env bash

models=(
  # # ResNet 계열
  # resnet18
  # resnet34
  # resnet50
  # resnet152

  # # densnet
  # densenet121
  # densenet161
  # densenet169
  # densenet201
  # densenet264

  # # EfficientNet B0~B7
  # efficientnet_b0
  # efficientnet_b1
  # efficientnet_b2
  # efficientnet_b3
  # efficientnet_b4
  # efficientnet_b5
  # efficientnet_b6
  # efficientnet_b7

  # # vgg
  # vgg11
  # vgg11_bn
  # vgg13
  # vgg13_bn
  # vgg16
  # vgg16_bn
  # vgg19
  # vgg19_bn

  # # vit 계열
  # vit_b_16
  # vit_b_32
  # vit_l_16
  # vit_l_32

  # swin 계열
  swin_b
  # swin_s
  # swin_t
  # swin_v2_b
  # swin_v2_s
  # swin_v2_t
)

for m in "${models[@]}"; do
  echo "▶▶▶ Training with MODEL_NAME=$m"
  MODEL_NAME=$m python3 train.py --cfg config.py || true
  echo
done
