#!/bin/bash
ROOT_DIR="/kaggle/input/adversarial-attack-image-eatmap-mask-epsilon03/output_images"
OUT_DIR="/kaggle/working"
EVAL_MODEL_DIR="/kaggle/working/thesis_2025/evaluate"

python3 data_transfer_eval.py --root_dir "$ROOT_DIR" --out_dir "$OUT_DIR" -- eval_model_dir "$EVAL_MODEL_DIR" --gpu 0