python compute_FR.py \
  --actual-dir "/kaggle/working/original" \
  --target-dir "/kaggle/working/adversarial" \
  --weights "/kaggle/input/resnet50-multilabel-model-add-layers/resnet50_multilabel_model_add_layers.pth" \
  --model "resnet50" \
  --batch-size 10 \
  --query-label 31 \
  --output-dir "/kaggle/working" \
  --gpu 0