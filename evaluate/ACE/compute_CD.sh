!python /kaggle/working/ACE/compute_CD.py \
    --oracle-path "/kaggle/working/ACE/models/checkpoint.tar" \
    --gpu 0 \
    --actual-path "/kaggle/working/original" \
    --output-path "/kaggle/working/adversarial" \
    --celeba-path "/kaggle/input/celebamaskhq/CelebAMask-HQ" \
    --dataset "CelebAHQ" \
    --query-label 31