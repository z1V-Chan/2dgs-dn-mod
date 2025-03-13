#!/bin/bash

# Dataset configuration
DATASET_PATH="data"
DATASET="Replica-SLAM-colmap"
OUTPUT_DATASET="replica_v"
SCENES=("room0" "room1" "room2")
# SCENES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")

# Common parameters
CUDA_DEVICE=7
RESOLUTION=1

for scene in "${SCENES[@]}"; do
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    python train.py \
        -s "$DATASET_PATH/$DATASET/$scene" \
        -m "output/$OUTPUT_DATASET/$scene" \
        -r $RESOLUTION \
        --test_iterations -1 \
        --depth_from_iter 40000 \
        --eval
done