#!/bin/bash

weights_path="/kaggle/input/resnet50-multilabel-model-add-layers/resnet50_multilabel_model_add_layers.pth"
output_dir="/kaggle/working/attack_results"

mkdir -p "$output_dir"

for i in {0..299}
do
  image="/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img/${i}.jpg"
  mask="/kaggle/input/heatmap-mask/kaggle/working/mask/results_${i}.png"
  sample_output="${output_dir}/sample_${i}"

  echo "Running attack on sample ${i}"

  python3 file_single.py \
    --weights "$weights_path" \
    --image "$image" \
    --mask "$mask" \
    --output_dir "$sample_output"
done