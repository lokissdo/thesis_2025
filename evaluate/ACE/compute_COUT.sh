python compute_COUT.py \
  --actual-dir "/kaggle/working/original" \
  --target-dir "/kaggle/working/adversarial" \
  --model-path "/kaggle/input/resnet50-multilabel-model-add-layers/resnet50_multilabel_model_add_layers.pth" \
  --model "resnet50" \
  --actual-label 0 \
  --target-label 31 \
  --gpu 0