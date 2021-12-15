BASE_DIR="../.."
DATASET_DIR="$BASE_DIR/dataset/multi_platform"

mkdir log
mkdir checkpoints
PID=10

for TRAIN_NUM in 32 100 200 300 -1; do
BASE=plt_${PID}_${TRAIN_NUM}_raw

python $BASE_DIR/predictor/main.py \
    --gpu 0 \
    --lr 0.001 \
    --steps 500 \
    --epochs 500 \
    --batch_size 16 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/gt.txt" \
    --train_test_stage \
    --train_num ${TRAIN_NUM} \
    --multi_plt ${PID} \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/$BASE.log" \
    --model_dir "checkpoints/$BASE" \
    --gnn_layer "SAGEConv" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --resume "checkpoints/$BASE/ckpt_best.pth" \
    --only_test \
    #--override_data
done
