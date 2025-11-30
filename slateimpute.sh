#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=0

DATASETS=("Cornell" "Texas" "Wisconsin" "Pubmed" "Cora" "Citeseer" "PPI" "CS")

GROUP_SIZES=(10)
MR=0.6

for dataset in "${DATASETS[@]}"; do
  echo "Running ours_1122 on dataset=${dataset}"
  for group_size in "${GROUP_SIZES[@]}"; do
    echo "  group_size=${group_size}, mr=${MR}"
    python -u slateimpute.py \
      --dataset "${dataset}" \
      --group_size "${group_size}" \
      --mr "${MR}" \
      --device "cuda:0" \
      "$@"
  done
done
