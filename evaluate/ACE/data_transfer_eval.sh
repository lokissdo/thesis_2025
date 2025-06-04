#!/bin/bash
ROOT_DIR="/kaggle/input/adversarial-attack-image-saliency-blended/output_images"
OUT_DIR="/kaggle/working/thesis_2025/evaluate"

python3 data_transfer_eval.py --root_dir "$ROOT_DIR" --out_dir "$OUT_DIR"
