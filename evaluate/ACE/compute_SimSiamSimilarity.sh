OUT_DIR="/kaggle/working"

python compute_SimSiamSimilarity.py \
    --weights-path "${OUT_DIR}/checkpoint_0099.pth.tar" \
    --actual-dir "${OUT_DIR}/original" \
    --target-dir "${OUT_DIR}/adversarial" \
    --gpu 0