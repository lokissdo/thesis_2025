OUT_DIR="/kaggle/working"
MODEL_EVAL_DIR="/kaggle/working/thesis_2025/evaluate/ACE/pretrained_models"
python compute_FVA.py \
    --cf-folder "${OUT_DIR}/adversarial" \
    --weights-path "${MODEL_EVAL_DIR}/resnet50_ft_weight.pkl" \
    --gpu 0 \
    --batch-size 15