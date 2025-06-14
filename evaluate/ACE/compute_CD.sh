OUT_DIR="/kaggle/working"
MODEL_EVAL_DIR="/kaggle/working/thesis_2025/evaluate/ACE/models"
python compute_CD.py \
    --oracle-path "${MODEL_EVAL_DIR}/checkpoint.tar" \
    --gpu 0 \
    --actual-path "${OUT_DIR}/original" \
    --output-path "${OUT_DIR}/adversarial" \
    --celeba-path "/kaggle/input/celebamaskhq/CelebAMask-HQ" \
    --dataset "CelebAHQ" \
    --query-label 31