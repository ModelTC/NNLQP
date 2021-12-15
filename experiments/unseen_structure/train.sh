BASE_DIR="../.."
DATASET_DIR="$BASE_DIR/dataset/unseen_structure"

mkdir log
mkdir checkpoints

for TEST_MODEL_TYPE in 'resnet18' 'vgg16' 'efficientb0' 'mobilenetv2' 'mobilenetv3' 'mnasnet' 'alexnet' 'squeezenet' 'googlenet' 'nasbench201';do

python $BASE_DIR/predictor/main.py \
    --gpu 0 \
    --lr 0.001 \
    --steps 500 \
    --epochs 50 \
    --batch_size 16 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/gt.txt" \
    --test_model_type ${TEST_MODEL_TYPE}\
    --norm_sf \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/$TEST_MODEL_TYPE.log" \
    --model_dir "checkpoints/$TEST_MODEL_TYPE" \
    --gnn_layer "SAGEConv" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    #--resume "checkpoints/$TEST_MODEL_TYPE/ckpt_best.pth" \
    #--only_test \
    #--override_data

done
