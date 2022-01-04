### Latency Measurement on FPGA

##### Environment
* Hardware: **Avnet Ultra96-V2 Board**
* Software: **Vitis-AI 1.4.0**
* Datatype: **int8**

##### Step
* Change onnx model to caffe model
* Quantize model by `vai_q_caffe`
* Compile and get xmodel by `vai_c_caffe`
* Run xmodel on DPU

##### Note
The following errors will occur during network measurement on FPGA, so the `gt.txt` file does not contain a large number of latency items.

| stage   | related networks | error | reason |
| :-----: | :-------:| :------: | :------: |
| compilation | EfficientNets & MobileNetV3s | ValueError: Unsupported op: type: sigmoid, name: xxxxx | not support Sigmoid |
| compilation | GoogleNets | [UNILOG][FATAL][XCOM_DATA_OUTRANGE][Data value is out of range!] | Unknown |
| compilation | MobileNetV2s | ValueError: Unsupported group convolution: group=xxxxx | not support group != input channel for Depthwise Conv|
| inference | NASBench201s & MNasNets | Check failed: round_mode == "DPU_ROUND"(STD_ROUND vs. DPU_ROUND) TODO | not support STD_ROUND |
| inference | ResNet18s & SqueezeNets | Check failed: handle != NULL cannot open library! lib=libvart_op_imp_conv2d-fix.so;error=libvart_op_imp_conv2d-fix.so: cannot open shared object file: No such file or directory;op=xir::Op{name = xxxxx, type = conv2d-fix} | not support kernel size >= 9 x 9  for Conv |
| inference | Vggs & AlexNets | Check failed: handle != NULL cannot open library! lib=libvart_op_imp_conv2d-fix.so;error=libvart_op_imp_conv2d-fix.so: cannot open shared object file: No such file or directory;op=xir::Op{name = xxxxx(TransferMatMulToConv2d), type = conv2d-fix} | not support input channel or output channel > 3072 for FC|