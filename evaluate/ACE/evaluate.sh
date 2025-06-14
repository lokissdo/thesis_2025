#!/bin/bash

echo "Initializing evaluation script..."
bash data_transfer_eval.sh

echo "Calculating COUT..."
bash compute_COUT.sh

echo "Calculating sFID..."
bash compute_sFID.sh

echo "Calculating Correlation Difference..."
bash compute_CD.sh

echo "Calculating FVA..."
bash compute_FVA.sh

echo "Calculating MNAC..."
bash compute_MNAC.sh

echo "Calculating FID..."
bash compute_FID.sh

echo "Calculating SimSiam Similarity..."
bash compute_SimSiamSimilarity.sh

echo "Calculating FVA..."
bash compute_FVA.sh

echo "Calculating Flip Ratio..."
bash compute_FR.sh
