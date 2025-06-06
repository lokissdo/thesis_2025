OUT_DIR="/kaggle/working"
python "${OUT_DIR}/ACE/compute_CD.py" \
    --oracle-path "${OUT_DIR}/ACE/models/checkpoint.tar" \
    --gpu 0 \
    --actual-path "${OUT_DIR}/original" \
    --output-path "${OUT_DIR}/adversarial" \
    --celeba-path "/kaggle/input/celebamaskhq/CelebAMask-HQ" \
    --dataset "CelebAHQ" \
    --query-label 31