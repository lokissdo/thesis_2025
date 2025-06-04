pip install pytorch-fid
REAL_PATH=/kaggle/working/original
CF_PATH=/kaggle/working/adversarial
python -m pytorch_fid ${REAL_PATH} ${CF_PATH}