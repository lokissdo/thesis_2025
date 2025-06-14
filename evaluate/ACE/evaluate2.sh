#!/bin/bash

echo "Initializing evaluation script..."
bash data_transfer_eval.sh

echo "Calculating Correlation Difference..."
bash compute_CD.sh

echo "Calculating SimSiam Similarity..."
bash compute_SimSiamSimilarity.sh