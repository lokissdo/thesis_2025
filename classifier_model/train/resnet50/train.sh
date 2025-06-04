#!/bin/bash

ROOT_DIR="/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img"
ATTR_FILE="/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=0.001

python3 train.py \
  --root_dir "$ROOT_DIR" \
  --attr_file "$ATTR_FILE" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE