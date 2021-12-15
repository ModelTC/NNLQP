BASE_DIR="../.."
DATASET_DIR="$BASE_DIR/dataset/multi_platform"

mkdir log
mkdir checkpoints

PID="plt_2_9_10_12_13_14_16_18_23"

python $BASE_DIR/predictor/main.py \
    --gpu 0 \
    --lr 0.001 \
    --steps 500 \
    --epochs 500 \
    --batch_size 16 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/gt.txt" \
    --train_test_stage \
    --multi_plt 2,9,10,12,13,14,16,18,23 \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/$PID.log" \
    --model_dir "checkpoints/$PID" \
    --gnn_layer "SAGEConv" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --resume "checkpoints/$PID/ckpt_best.pth" \
    --only_test \
    #--override_data
