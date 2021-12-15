# NNLQP

NNLQP: A Multi-Platform Neural Network Latency Query and Prediction System with An Evolving Database

## Installation

#### Environment
  * `Ubuntu 16.04`
  * `python 3.6.4`

#### Install pytorch onnx networkx
  * `pytorch==1.5.0`
  * `onnx==1.7.0`
  * `networkx==2.5.1`
```shell
pip3 install -r requirements.txt
```

#### Install torch-geometric
```shell
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.5.0%2Bcpu/torch_scatter-2.0.5-cp36-cp36m-linux_x86_64.whl
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.5.0%2Bcpu/torch_sparse-0.6.7-cp36-cp36m-linux_x86_64.whl
pip3 install torch-cluster -f https://data.pyg.org/whl/torch-1.5.0%2Bcpu/torch_cluster-1.5.7-cp36-cp36m-linux_x86_64.whl
pip3 install torch-geometric
```

#### Run Demo
Clone the repository
```shell
git clone https://github.com/anonymousnnlqp/NNLQP.git
```

Given an onnx model as input, the demo can predict model latencies on 9 platforms:
```shell
cd nnlqp
python3 demo.py --onnx_file test_onnx/resnet18-v1-7-no-weight.onnx
```

Output:
```text
Read Onnx: test_onnx/resnet18-v1-7-no-weight.onnx
Model inference cost: 44.1431999206543 ms
Latency prediction for platform cpu-openppl-fp32 : 386.098876953125 ms
Latency prediction for platform hi3559A-nnie11-int8 : 62.860267639160156 ms
Latency prediction for platform gpu-T4-trt7.1-fp32 : 7.964828014373779 ms
Latency prediction for platform gpu-T4-trt7.1-int8 : 1.4778786897659302 ms
Latency prediction for platform gpu-P4-trt7.1-fp32 : 9.099410057067871 ms
Latency prediction for platform gpu-P4-trt7.1-int8 : 3.7260262966156006 ms
Latency prediction for platform hi3519A-nnie12-int8 : 61.44997787475586 ms
Latency prediction for platform atlas300-acl-fp16 : 9.21658992767334 ms
Latency prediction for platform mul270-neuware-int8 : 18.250690460205078 ms
```

## Dataset

#### Download
```shell
cd nnlqp
mkdir dataset
wget https://github.com/anonymousnnlqp/NNLQP/releases/download/v1.0-data/dataset.tar.gz -O dataset.tar.gz
tar -xzvf dataset.tar.gz -C dataset
```
#### Format
The latency dataset is saved to directory `nnlqp/dataset`:
```text
└── dataset
    ├── multi_platform
    │   ├── gt.txt
    │   └── onnx
    │       ├── ...
    │       ├── ...
    └── unseen_structure
        ├── gt.txt
        └── onnx
            ├── ...
            ├── ...
```

`onnx` is the directory of onnx models, which are removed weights.
`gt.txt` is the latency ground-truth file, each line of the file is format as:

```text
${GraphID} ${OnnxPath} ${BatchSize} ${Latency}(ms) ${ModelType} ${PlatformId} [${Stage}](train/test)
```

The `gt.txt` file can like:
```text
553312 onnx/nnmeter_alexnet/nnmeter_alexnet_transform_0130.onnx 8 90.88183 alexnet 2 train
559608 onnx/nnmeter_alexnet/nnmeter_alexnet_transform_0157.onnx 8 11.1463 alexnet 23 train
549409 onnx/nnmeter_alexnet/nnmeter_alexnet_transform_0197.onnx 8 64.11547 alexnet 16 test
...
```
#### Overview
The datasets contains two parts: `unseen strucutre` and `multi platform`.
We will constantly add new structures and platforms to the dataset.
##### Unseen Structure
  * Platforms
    * `gpu-gtx1660-trt7.1-fp32`
  * Onnx models (2000 * 10)
    * [x] ResNets
    * [x] VGGs
    * [x] EfficientNets
    * [x] MobileNetV2s
    * [x] MobileNetV3s
    * [x] MnasNets
    * [x] AlexNets
    * [x] SqueezeNets
    * [x] GoogleNets
    * [x] NasBench201s
    * [x] VisionTransformers
  * Input sizes
    * `1 x 3 x 32 x 32` (only for NasBenc201s)
    * `1 x 3 x 224 x224`
  * Latency samples
    * 20000

##### Multi Platform
  * Platforms
    * [x] cpu-openppl-fp32 (CPU)
    * [x] hi3559A-nnie11-int8 (NPU)
    * [x] gpu-T4-trt7.1-fp32 (GPU)
    * [x] gpu-T4-trt7.1-int8 (GPU)
    * [x] gpu-P4-trt7.1-fp32 (GPU)
    * [x] gpu-P4-trt7.1-int8 (GPU)
    * [x] hi3519A-nnie12-int8 (NPU)
    * [x] atlas300-acl-fp16 (NPU)
    * [x] mul270-neuware-int8 (NPU)
    * [ ] hexagonDSP-snpe-int8 (DSP)
  * Onnx models (200 * 10)
    * ResNets
    * VGGs
    * EfficientNets
    * MobileNetV2s
    * MobileNetV3s
    * MnasNets
    * AlexNets
    * SqueezeNets
    * GoogleNets
    * NasBench201s
  * Input sizes
    * `8 x 3 x 32 x 32` (only for NasBenc201s)
    * `8 x 3 x 224 x224`
  * Latency samples
    * 10597

## Experiments

#### Unseen Structure
We use a certain model type for test, and use other types for training. The user can specify the test model type by specifying the `TEST_MODEL_TYPE` in the script.

Before experiments, please make sure that you download the dataset, and the directory `nnlqp/dataset/unseen_structure/` are valid.

```shell
cd nnlqp/experiments/unseen_structure/
```

* Get our pre-trained models and test
```shell
wegt https://github.com/anonymousnnlqp/NNLQP/releases/download/v1.0-data/unseen_structure_ckpt.tar.gz -O unseen_structure_ckpt.tar.gz
tar -xzvf unseen_structure_ckpt.tar.gz
bash test.sh
```

* Train from the beginning
```shell
bash train.sh
```

#### Multi Platform

Before experiments, please make sure that you download the dataset, and the directory `nnlqp/dataset/multi_platform/` are valid.

```shell
cd nnlqp/experiments/multi_platform/
```

* Get our pre-trained models and test
```shell
wget https://github.com/anonymousnnlqp/NNLQP/releases/download/v1.0-data/multi_platform_ckpt.tar.gz -O multi_platform_ckpt.tar.gz
tar -xzvf multi_platform_ckpt.tar.gz
bash test.sh
```

* Train from the beginning
Users can change `--multi_plt=1,3,4` to train a predictor that predicts latency for platforms 1, 3, 4 at the same time. The platform id should have appeared in the file `gt.txt`

```shell
bash train.sh
```

#### Transfer Platform

If there are latency samples on 4 platforms A, B, C, D, we can first train a pre-trained model that predicts latency on platforms A, B, C. Then we can adopt the pre-trained model to initialize weights, and use a small count of latency samples to train a predictor fo platform D. The knowledge of the model structure learned by the predictor for platforms A, B, C can be transferred to platform D.

For example, the `multi platforms` dataset involved 9 platforms, we can train a pre-trained model on 8 platforms 2, 9, 12, 13, 14, 16, 18, 23, and then use a certain number of samples to train predictor of platform 10. Users can set `TRAIN_NUM` in `raw_train.sh` or `transfer_train.sh` to control the samples numbers. In this example, we use `32`, `100`, `200`, `300`, `-1 (all)`. `raw_train.sh` is for training a predictor without the pre-trained model and transfer learning, while `transfer_train.sh` uses the pre-trained model and transfer learning.

```shell
cd nnlqp/experiments/transfer_platform/
```

* Users can test the results of trained predictors:
```shell
wget https://github.com/anonymousnnlqp/NNLQP/releases/download/v1.0-data/transfer_platform_ckpt.tar.gz -O transfer_platform_ckpt.tar.gz
tar -xzvf transfer_platform_ckpt.tar.gz
bash raw_test.sh
bash transfer_test.sh
```

* Or just train from the beginning:
```shell
bash raw_train.sh
bash transfer_train.sh
```

#### Custom Predictor

First, users should prepare the latency dataset with [the same format as our dataset](#format).

Then you can train your own latency predictor in two different ways:

* Train from the beginning:
  * refer the script `nnlqp/unseen_structure/train.sh`

* Transfer from our pre-trained model (more suitable for a small count of latency samples)
  * refer the script `nnlqp/transfer_platform/transfer_train.sh`