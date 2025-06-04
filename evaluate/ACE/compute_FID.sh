pip install pytorch-fid
REAL_PATH=/kaggle/working/thesis_2025/evaluate/original
CF_PATH=/kaggle/working/thesis_2025/evaluate/adversarial
python -m pytorch_fid ${REAL_PATH} ${CF_PATH}