OUT_DIR="/kaggle/working"
MODEL_EVAL_DIR="/kaggle/working/thesis_2025/evaluate/ACE/models"
python compute_MNAC.py \
    --oracle-path "${MODEL_EVAL_DIR}/checkpoint.tar" \
    --actual-dir "${OUT_DIR}/original" \
    --target-dir "${OUT_DIR}/adversarial" \
    --gpu 0