# --- run_attack.sh ---
#!/bin/bash

weights_path="/kaggle/input/resnet50-multilabel-model-add-layers/resnet50_multilabel_model_add_layers.pth"
output_dir="/kaggle/working/attack_results"

mkdir -p "$output_dir"

for i in {0..299}
do
  original="/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img/${i}.jpg"
  input="/kaggle/input/sd2-new-prompt-clip-classifier/edit_results/sample_${i}/res_1.jpg"
  mask="/kaggle/input/heatmap-mask/kaggle/working/mask/results_${i}.png"
  sample_output="${output_dir}/sample_${i}"

  echo "Running attack on sample ${i}"
  mkdir -p "$sample_output"

  python3 run_attack_cli.py \
    --weights "$weights_path" \
    --original "$original" \
    --input "$input" \
    --mask "$mask" \
    --output_dir "$sample_output" \
    --attack_method "pgd"
done