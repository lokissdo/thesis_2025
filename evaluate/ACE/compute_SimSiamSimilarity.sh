OUT_DIR="/kaggle/working"
MODEL_EVAL_DIR="/kaggle/working/thesis_2025/evaluate/ACE/pretrained_models"
python compute_SimSiamSimilarity.py \
    --weights-path "${MODEL_EVAL_DIR}/checkpoint_0099.pth.tar" \
    --actual-dir "${OUT_DIR}/original" \
    --target-dir "${OUT_DIR}/adversarial" \
    --gpu 0