# QDQ ONNX scale 0 to 1 converter

This tool convert QuantizeLinear/DequantizeLinear's scale 0 to 1.
In particular, the tool performs the following steps:

1. constant folding
2. convert scale 0 to 1

## Motivation
If QuantizeLinear/DequantizeLinear has all 0 scale when trying to convert QDQ ONNX to TensorRT's engine, trtexec command report following error.

> [E] [TRT] ModelImporter.cpp:731: ERROR: builtin_op_importers.cpp:1216 In function QuantDequantLinearHelper:[6] Assertion failed: scaleAllPositive && "Scale coefficients must all be positive"

I guess all 0 scales become from all 0 tensor in the calibration stage.
For example, convolution layer outputs negative values and following Relu layer outputs all 0, and become quantized scale to 0.

Because of all 0 tensor, scale value is OK in any values.(0 * any == 0)
So, this tool convert scale 0 to 1.
(If scale is 1, trtexec converts successfully.)

## Dependency
* numpy
* onnx
* onnxruntime
* [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)

## Installation
```bash
pip3 install numpy onnx onnxruntime
pip3 install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
pip3 install git+https://github.com/maminus/qdqonnx_scale_0to1.git
```

## Usage
```bash
qdq_scale0to1 -i original.onnx -o converted.onnx
```

###### caution
If it does not exists in the `PATH` environment variable, add tool's path(e.g. `~/.local/bin`) to `PATH`.
