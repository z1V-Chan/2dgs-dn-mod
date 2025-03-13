DATASET_PATH="data"
DATASET="Replica-SLAM-colmap"
# SCENES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")
OUTPUT_DATASET="replica"
SCENES=("office0" "office1" "office2")

for scene in "${SCENES[@]}"; do
    echo "Processing: $scene"
    # OMP_NUM_THREADS=128 CUDA_VISIBLE_DEVICES=0 python render.py -m "outputs/$DATASET/$scene" -s "datasets/$DATASET/$scene"
    # OMP_NUM_THREADS=128 CUDA_VISIBLE_DEVICES=2 python render.py -m "outputs/$DATASET/$scene" -s "$DATASET_PATH/$DATASET/$scene" --voxel_size 0.03
    OMP_NUM_THREADS=128 CUDA_VISIBLE_DEVICES=6 python render.py \
        -m "output/$OUTPUT_DATASET/$scene" \
        -s "$DATASET_PATH/$DATASET/$scene" \
        --depth_trunc 5 \
        --voxel_size 0.03 \
        --skip_mesh
done
