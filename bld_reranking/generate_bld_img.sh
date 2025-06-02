python generate_bld_img.py \
  --img_dir "/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebA-HQ-img" \
  --attr_file "/kaggle/input/celebamaskhq/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt" \
  --mask_dir "/kaggle/input/mask-smiling-kltn/MaskSmiling" \
  --model_path "/kaggle/input/resnet50-multilabel-model-add-layers/resnet50_multilabel_model_add_layers.pth" \
  --out_dir "/kaggle/working/edit_results" \
  --attr_idx 31 \
  --start 0 \
  --end 300