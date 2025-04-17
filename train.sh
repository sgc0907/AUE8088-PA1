#!/usr/bin/env bash
set -e

models=(
  # ResNet 계열
  # resnet18
  # resnet34
  # resnet50
  # resnet152
  densenet201

  # EfficientNet B0~B7
  efficientnet_b0
  efficientnet_b1
  efficientnet_b2
  efficientnet_b3
  efficientnet_b4
  efficientnet_b5
  efficientnet_b6
  efficientnet_b7
)

for m in "${models[@]}"; do
  echo "▶▶▶ Training with MODEL_NAME=$m"
  MODEL_NAME=$m python3 train.py
  echo
done
