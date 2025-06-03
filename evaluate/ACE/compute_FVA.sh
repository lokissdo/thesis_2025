OUT_DIR="/kaggle/working"
python compute_FVA.py \
    --cf-folder "${OUT_DIR}/adversarial" \
    --weights-path "${OUT_DIR}/resnet50_ft_weight.pkl" \
    --gpu 0 \
    --batch-size 15