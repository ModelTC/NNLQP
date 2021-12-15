import onnx
import logging
import numpy as np
from .onnx_shape_infer import custom_shape_infer, onnx_node_attributes_to_dict


def flops_conv(node, attrs, maps):
    assert(node.op_type == 'Conv')
    input_shape = maps[node.input[1]]
    output_shape = maps[node.output[0]]

    Co_Ho_Wo = np.prod(output_shape[1:])
    ci_k2 = np.prod(input_shape[1:]) # cin // group * k * k
    # group = attrs['group'] if 'group' in attrs else 1
    bias = 1 if len(node.input) == 3 else 0

    # flops: Co * Ho * Wo * (2 * Ci * k * k / group - 1 + bias)
    # params: Co * (Ci * k * k + bias)
    flops = Co_Ho_Wo * (2 * ci_k2 - 1 + bias)
    params = input_shape[0] * (np.prod(input_shape[1:]) + bias)
    return flops, params


def flops_gemm(node, attrs, maps):
    assert(node.op_type == 'Gemm')
    input_shape = maps[node.input[1]]
    if 'transB' in attrs and attrs['transB'] != 0:
        Co, Ci = input_shape[-2:]
    else:
        Ci, Co = input_shape[-2:]
    bias = 1 if len(node.input) == 3 else 0

    # flops: Co * (2 * Ci - 1 +bias)
    # params: Co * (Ci + bias)
    flops = Co * (2 * Ci - 1 + bias)
    params =  Co * (Ci + bias)
    return flops, params


def flops_matmal(node, attrs, maps):
    assert(node.op_type == 'MatMul')
    # (N x K) * (K x M) = (N x M)
    N, K = maps[node.input[0]][-2:]
    N, M = maps[node.output[0]][-2:]

    # flops: N * M * (2 * K - 1)
    # params: 0, because there not existed fixed weight
    flops = N * M * (2 * K - 1)
    params = 0
    return flops, params


def flops_avgpool(node, attrs, maps):
    assert node.op_type == 'GlobalAveragePool' or node.op_type == 'AveragePool'
    if node.op_type == 'GlobalAveragePool':
        kernel_shape = maps[node.input[0]][3] / maps[node.output[0]][3]
    else:
        kernel_shape = attrs['kernel_shape']
    output = maps[node.output[0]]
    flops = np.prod(output[1:]) * np.prod(kernel_shape)
    params = 0
    return flops, params


def flops_zero(node, attrs, maps):
    return 0, 0


def flops_add(node, attrs, maps):
    flops = np.prod(maps[node.output[0]])
    return flops, 0


def flops_maxpool(node, attrs, maps):
    return 0, 0


def flops_softmax(node, attrs, maps):
    x = np.array(maps[node.input[0]])
    nfeatures = x.size // x[0]

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    flops = 1 * (total_exp + total_add + total_div)
    return flops, 0


def flops_bn(node, attrs, maps):
    x = maps[node.input[0]]
    if len(node.input) > 1:
        flops = 4 * np.prod(x)
        return flops, 4*x[1]
    else:
        flops = 2 * np.prod(x)
        return flops, 2*x[1]


def flops_upsample(node, attrs, maps):
    if attrs['mode'] not in (b"nearest", b"linear", b"bilinear", b"bicubic", b"trilinear"):
        logging.warning("mode %s is not implemented yet, take it a zero op" % attrs['mode'])
        return flops_zero(node, attrs, maps)

    if attrs['mode'] == b"nearest":
        return flops_zero(node, attrs, maps)

    y = maps[node.output[0]]
    if attrs['mode'] == b"linear":
        flops = np.prod(y[1:]) * 5  # 2 muls + 3 add
    elif attrs['mode'] == b"bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = np.prod(y[1:]) * 11  # 6 muls + 5 adds
    elif attrs['mode'] == b"bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        flops = np.prod(y[1:]) * (ops_solve_A + ops_solve_p)
    elif attrs['mode'] == b"trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        flops = np.prod(y[1:]) * (13 * 2 + 5)
    return flops, 0


def calculate_onnx_flops(onnx_G, each_node=False):
    flops = 0.0
    params = 0.0
    memory_access = 0.0
    node_flops = {}

    status, G, output_shapes = custom_shape_infer(onnx_G)

    flops_funcs = {
        "Conv": flops_conv,
        "Relu": flops_zero,
        "Add": flops_add,
        "Sigmoid": flops_add,
        "Reshape": flops_zero,
        "MaxPool": flops_zero,
        "Split": flops_zero,
        "GlobalAveragePool": flops_avgpool,
        "Gemm": flops_gemm,
        "Transpose": flops_zero,
        "Upsample": flops_upsample,
        "BatchNormalization": flops_bn,
        "Mul": flops_add,
        "Concat": flops_zero,
        "Flatten": flops_zero,
        "AveragePool": flops_avgpool,
        "Cast": flops_zero,
        "Matmul": flops_matmal,
        "ReduceMean": flops_add,
        "Pow": flops_add,
        "Slice": flops_zero,
        "Div": flops_add,
        "Sub": flops_add,
        "Sqrt": flops_add,
        "Clip": flops_zero,
        "Softmax": flops_softmax,
        "Tanh": flops_add,
        "ConvTranspose": flops_conv
    }

    # caculate flops node by node
    if status is True:
        for node in G.graph.node:
            cur_access = 0
            cur_params = 0
            cur_flops = 0

            for output in node.output:
                # memory_access = feature_map_sizes + params
                cur_access += np.prod(output_shapes[output])

            if node.op_type in flops_funcs:
                attrs = onnx_node_attributes_to_dict(node.attribute)
                cur_flops, cur_params = flops_funcs[node.op_type](node, attrs, output_shapes)
                flops += cur_flops
                params += cur_params
                cur_access += cur_params
                memory_access += cur_access

            for output in node.output:
                node_flops[output] = (cur_flops, cur_params, cur_access)

    if each_node is True:
        return flops, params, memory_access, node_flops
    else:
        return flops, params, memory_access


if __name__ == '__main__':
    import sys
    onnx_path = sys.argv[1]
    onnx_G = onnx.load(onnx_path)
    print(calculate_onnx_flops(onnx_G, each_node=False))
