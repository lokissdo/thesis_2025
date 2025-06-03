OUT_DIR="/kaggle/working"

python compute_MNAC.py \
    --oracle-path "${OUT_DIR}/checkpoint.tar" \
    --actual-dir "${OUT_DIR}/original" \
    --target-dir "${OUT_DIR}/adversarial" \
    --gpu 0