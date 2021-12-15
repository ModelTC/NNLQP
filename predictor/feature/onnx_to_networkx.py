import os
import onnx
import copy
import logging
import networkx as nx

from onnx.numpy_helper import to_array
from .op_attribute import *


# construct networkx graph from onnx file
def onnx2nx(onnx_path):
    global WARNINGS
    WARNINGS.clear()

    if isinstance(onnx_path, str):
        assert(os.path.exists(onnx_path))
        switch_print("Read Onnx: {}".format(onnx_path))
        onnx_G = onnx.load(onnx_path)
    else:
        assert isinstance(onnx_path, onnx.onnx_ml_pb2.ModelProto), "onnx2nx input should be str or ModelProto"
        onnx_G = onnx_path

    nx_G = nx.DiGraph()
    switcher = Switcher(onnx_G)

    all_G_edges = []
    all_G_nodes = []
    onnx_nodes = onnx_G.graph.node

    for node in onnx_nodes:
        G_nodes, G_edges = switcher.parse_node(node)

        # collect the node and edges
        if G_nodes is not None:
            all_G_nodes.extend(G_nodes)
            all_G_edges.extend(G_edges)

    # add the node & edges to networkx graph
    nx_G.add_nodes_from(all_G_nodes)
    nx_G.add_edges_from(all_G_edges)

    input_sizes = {}
    output_sizes = switcher.output_sizes

    # for input nodes
    zero_indegree = [v for v, d in nx_G.in_degree() if d == 0]
    for node in zero_indegree:
        if nx_G.out_degree()[node] > 0 and node in switcher.input_sizes:
            nx_G.add_nodes_from([(node, {'attr': AttrInput(name=node)})])
            input_sizes[node] = switcher.input_sizes[node]
        else:
            # zero input, zero output, delete the node
            nx_G.remove_node(node)

    if len(WARNINGS) > 0:
        path_name = onnx_path.split('/')[-1] if isinstance(onnx_path, str) else "ModelProto"
        switch_print("[{}] onnx -> networkx warnings: ".format(path_name))
        for w in WARNINGS:
            switch_print(" -- {}".format(w))

    return PGraph(nx_G, input_sizes, output_sizes, switcher.opsets, onnx_G)


class PGraph:
    def __init__(self, G, input_sizes, output_sizes, opsets, onnx_G):
        self.data = G
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.opsets = opsets
        self.onnx_G = onnx_G


def switch_print(info):
    logger = logging.getLogger('GPDB')
    print(info)
    logger.info(info)


class Switcher:
    def __init__(self, onnx_G):
        self.onnx_G = onnx_G
        self.input_sizes = {}
        self.output_sizes = {}

        # opset info
        opset_cnt = len(self.onnx_G.opset_import)
        if opset_cnt <= 0:
            self.opsets = [9]
        else:
            self.opsets = [self.onnx_G.opset_import[x].version for x in range(opset_cnt)]

        # input and params info
        for inp in self.onnx_G.graph.input:
            self.input_sizes[inp.name] = tuple([x.dim_value for x in inp.type.tensor_type.shape.dim])
        # some weight is not regarded as input
        for init in self.onnx_G.graph.initializer:
            self.input_sizes[init.name] = tuple(init.dims)
        for out in self.onnx_G.graph.output:
            self.output_sizes[out.name] = tuple([x.dim_value for x in out.type.tensor_type.shape.dim])

    def parse_node(self, node):
        try:
            parse_func = getattr(self, "parse" + node.op_type)
        except Exception as e:
            raise Exception("{}, Operator [{}] Not Supported!".format(e, node.op_type))
        return parse_func(node)

    # parse onnx tensor value
    def parse_tensor(self, tensor, dims_only=False):
        dims = tuple(tensor.dims)
        if dims_only:
            return dims

        val = to_array(tensor)
        if len(dims) > 0:
            val = tuple(val)
        return val

    # parse onnx attribute of op node
    def parse_attrs(self, node_attrs, dims_only=False):
        attrs = {'opsets': self.opsets}
        for attr in node_attrs:
            if attr.type == onnx.AttributeProto.AttributeType.INTS:
                attrs[attr.name] = tuple(attr.ints)
            elif attr.type == onnx.AttributeProto.AttributeType.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
                attrs[attr.name] = tuple(attr.floats)
            elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
                attrs[attr.name] = self.parse_tensor(attr.t, dims_only)
            elif attr.type == onnx.AttributeProto.AttributeType.STRING:
                attrs[attr.name] = str(attr.s)
            elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
                attrs[attr.name] = tuple([str(x) for x in attr.strings])
            else:
                raise Exception("ATTR Type [{}] Not Supported!".format(attr.type))
        return attrs

    # Construct the node and edges of networkx graph
    def get_networkx_node_edges(self, node, G_attr, conn_input_ids, conn_output_ids):
        G_nodes = []
        if len(conn_output_ids) == 1:
            idx = conn_output_ids[0]
            G_nodes.append((node.output[idx], {'attr': G_attr}))

        else:
            # multi output
            for idx in conn_output_ids:
                G_attr_cur = copy.deepcopy(G_attr)
                G_attr_cur.attributes['output_num'] = len(conn_output_ids)
                G_attr_cur.attributes['output_idx'] = idx
                G_nodes.append((node.output[idx], {'attr': G_attr_cur}))

        G_edges = []
        for idx in conn_input_ids:
            for idy in conn_output_ids:
                G_edges.append((node.input[idx], node.output[idy]))

        return G_nodes, G_edges

    # assert the number of input and output
    def assert_node_input_output(self, node, input_nums, output_nums,
                                    input_low_bound=None, output_low_bound=None):
        in_num, out_num = len(node.input), len(node.output)

        if input_nums is not None and in_num not in input_nums:
            raise Exception("Input num of <{}> = {}, which are not in {}!". \
                            format(node.op_type, in_num, input_nums))

        if output_nums is not None and out_num not in output_nums:
            raise Exception("Output num of <{}> = {}, which are not in {}!". \
                            format(node.op_type, out_num, output_nums))

        if input_low_bound is not None and in_num < input_low_bound:
            raise Exception("Input num of <{}> = {}, which are not >= {}!". \
                            format(node.op_type, in_num, input_low_bound))

        if output_low_bound is not None and out_num < output_low_bound:
            raise Exception("Output num of <{}> = {}, which are not >= {}!". \
                            format(node.op_type, out_num, output_low_bound))

    # parse the general node without additional attributes
    def parse_general_node(self, node, AttrOp, conn_input_ids, conn_output_ids):
        attrs = self.parse_attrs(node.attribute)
        G_attr = AttrOp(name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, conn_output_ids)

    # ------------------------------------------------------------------------------------
    #                             For each onnx op type
    # ------------------------------------------------------------------------------------

    def parseAbs(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAbs, [0], [0])

    def parseAcos(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAcos, [0], [0])

    def parseAcosh(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAcosh, [0], [0])

    def parseAdd(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrAdd, [0, 1], [0])

    def parseAnd(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrAnd, [0, 1], [0])

    def parseArgMax(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrArgMax, [0], [0])

    def parseArgMin(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrArgMin, [0], [0])

    def parseAsin(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAsin, [0], [0])

    def parseAsinh(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAsinh, [0], [0])

    def parseAtan(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAtan, [0], [0])

    def parseAtanh(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAtanh, [0], [0])

    def parseAveragePool(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrAveragePool, [0], [0])

    def parseBatchNormalization(self, node):
        self.assert_node_input_output(node, [5], [1, 2, 3, 4, 5])
        attrs = self.parse_attrs(node.attribute)

        # Get the size of scale, B, mean and var
        # without tuple: Error: can't pickle repeated message fields, convert to list first
        conn_input_ids = [0, 1, 2, 3, 4]
        scale_size = B_size = mean_size = var_size = -1
        if node.input[1] in self.input_sizes:
            scale_size = self.input_sizes[node.input[1]]
            conn_input_ids.remove(1)
        if node.input[2] in self.input_sizes:
            B_size = self.input_sizes[node.input[2]]
            conn_input_ids.remove(2)
        if node.input[3] in self.input_sizes:
            mean_size = self.input_sizes[node.input[3]]
            conn_input_ids.remove(3)
        if node.input[4] in self.input_sizes:
            var_size = self.input_sizes[node.input[4]]
            conn_input_ids.remove(4)

        G_attr = AttrBatchNormalization(scale_size, B_size, mean_size, var_size,
                    name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, range(len(node.output)))

    def parseBitShift(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrBitShift, [0, 1], [0])

    def parseCast(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrCast, [0], [0])

    def parseCeil(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrCeil, [0], [0])

    def parseCelu(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrCelu, [0], [0])

    def parseClip(self, node):
        self.assert_node_input_output(node, [1, 2, 3], [1])
        return self.parse_general_node(node, AttrClip, range(len(node.input)), [0])

    def parseCompress(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrCompress, [0, 1], [0])

    def parseConcat(self, node):
        self.assert_node_input_output(node, None, [1], input_low_bound=1)
        return self.parse_general_node(node, AttrConcat, range(len(node.input)), [0])

    def parseConcatFromSequence(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrConcat, [0], [0])

    def parseConstant(self, node):
        self.assert_node_input_output(node, [0], [1])
        attrs = self.parse_attrs(node.attribute, dims_only=True)

        # Constant node could be weight param of conv node
        self.input_sizes[node.output[0]] = attrs['value']

        G_attr = AttrConstant(name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, [], [0])

    def parseConstantOfShape(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrConstantOfShape, [0], [0])

    def parseConv(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        attrs = self.parse_attrs(node.attribute)

        # Get input channel and output channel size
        conn_input_ids = [0]
        if node.input[1] in self.input_sizes:
            # param
            output_channel, input_channel = self.input_sizes[node.input[1]][:2]
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = self.input_sizes[node.input[1]][-2:]
        else:
            # weight as node
            output_channel, input_channel = -1, -1
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = -1
            conn_input_ids.append(1)
        bias = True if len(node.input) == 3 else False

        G_attr = AttrConv(input_channel, output_channel, bias, name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, [0])

    def parseConvInteger(self, node):
        self.assert_node_input_output(node, [2, 3, 4], [1])
        attrs = self.parse_attrs(node.attribute)

        # Get input channel and output channel size
        conn_input_ids = list(range(node.input))
        if node.input[1] in self.input_sizes:
            # param
            output_channel, input_channel = self.input_sizes[node.input[1]][:2]
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = self.input_sizes[node.input[1]][-2:]
            conn_input_ids.remove(1)
        else:
            # weight as node
            output_channel, input_channel = -1, -1
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = -1

        G_attr = AttrConvInteger(input_channel, output_channel, name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, [0])

    def parseConvTranspose(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        attrs = self.parse_attrs(node.attribute)

        # Get input channel and output channel size
        conn_input_ids = [0]
        if node.input[1] in self.input_sizes:
            # param
            output_channel, input_channel = self.input_sizes[node.input[1]][:2]
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = self.input_sizes[node.input[1]][-2:]
        else:
            # weight as node
            output_channel, input_channel = -1, -1
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = -1
            conn_input_ids.append(1)
        bias = True if len(node.input) == 3 else False

        G_attr = AttrConvTranspose(input_channel, output_channel, bias, name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, [0])

    def parseCos(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrCos, [0], [0])

    def parseCosh(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrCosh, [0], [0])

    def parseCumSum(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrCumSum, [0, 1], [0])

    def parseDepthToSpace(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrDepthToSpace, [0], [0])

    def parseDequantizeLinear(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        return self.parse_general_node(node, AttrDequantizeLinear, range(len(node.input)), [0])

    def parseDet(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrDet, [0], [0])

    def parseDiv(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrDiv, [0, 1], [0])

    def parseDropout(self, node):
        self.assert_node_input_output(node, [1, 2, 3], [1, 2])
        return self.parse_general_node(node, AttrDropout, range(len(node.input)), range(len(node.output)))

    def parseDynamicQuantizeLinear(self, node):
        self.assert_node_input_output(node, [1], [3])
        return self.parse_general_node(node, AttrDynamicQuantizeLinear, [0], [0, 1, 2])

    def parseEinsum(self, node):
        self.assert_node_input_output(node, None, [1], input_low_bound=1)
        return self.parse_general_node(node, AttrEinsum, range(len(node.input)), [0])

    def parseElu(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrElu, [0], [0])

    def parseEqual(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrEqual, [0, 1], [0])

    def parseErf(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrErf, [0], [0])

    def parseExp(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrExp, [0], [0])

    def parseExpand(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrExpand, [0, 1], [0])

    def parseEyeLike(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrEyeLike, [0], [0])

    def parseFlatten(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrFlatten, [0], [0])

    def parseFloor(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrFloor, [0], [0])

    def parseGRU(self, node):
        self.assert_node_input_output(node, [3, 4, 5, 6], [0, 1, 2])
        return self.parse_general_node(node, AttrGRU, range(len(node.input)), range(len(node.output)))

    def parseGather(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrGather, [0, 1], [0])

    def parseGatherElements(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrGatherElements, [0, 1], [0])

    def parseGatherND(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrGatherND, [0, 1], [0])

    def parseGemm(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        attrs = self.parse_attrs(node.attribute)

        # Get the input and output size
        conn_input_ids = [0]
        if node.input[1] in self.input_sizes:
            # params
            if 'transB' in attrs and attrs['transB'] != 0:
                output_size, input_size = self.input_sizes[node.input[1]][:2]
            else:
                input_size, output_size = self.input_sizes[node.input[1]][:2]
        else:
            # weight as node
            input_size, output_size = -1, -1
            conn_input_ids.append(1)
        bias = True if len(node.input) == 3 else False

        G_attr = AttrGemm(input_size, output_size, bias, name=node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, [0])

    def parseGlobalAveragePool(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrGlobalAveragePool, [0], [0])

    def parseGlobalLpPool(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrGlobalLpPool, [0], [0])

    def parseGlobalMaxPool(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrGlobalMaxPool, [0], [0])

    def parseGreater(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrGreater, [0, 1], [0])

    def parseGreaterOrEqual(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrGreaterOrEqual, [0, 1], [0])

    def parseHardSigmoid(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrHardSigmoid, [0], [0])

    def parseHardmax(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrHardmax, [0], [0])

    def parseIdentity(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrIdentity, [0], [0])

    def parseIf(self, node):
        self.assert_node_input_output(node, [1], None, output_low_bound=1)
        return self.parse_general_node(node, AttrIf, [0], range(len(node.output)))

    def parseInstanceNormalization(self, node):
        self.assert_node_input_output(node, [3], [1])
        attrs = self.parse_attrs(node.attribute)

        # Get the size of scale, B
        conn_input_ids = [0, 1, 2]
        scale_size = B_size = -1
        if node.input[1] in self.input_sizes:
            scale_size = self.input_sizes[node.input[1]]
            conn_input_ids.remove(1)
        if node.input[2] in self.input_sizes:
            B_size = self.input_sizes[node.input[2]]
            conn_input_ids.remove(2)

        G_attr = AttrInstanceNormalization(scale_size, B_size, name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, [0])

    def parseIsInf(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrIsInf, [0], [0])

    def parseIsNaN(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrIsNaN, [0], [0])

    def parseLRN(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrLRN, [0], [0])

    def parseLSTM(self, node):
        self.assert_node_input_output(node, [3, 4, 5, 6, 7, 8], [0, 1, 2, 3])
        return self.parse_general_node(node, AttrGRU, range(len(node.input)), range(len(node.output)))

    def parseLeakyRelu(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrLeakyRelu, [0], [0])

    def parseLess(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrLess, [0, 1], [0])

    def parseLessOrEqual(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrLessOrEqual, [0, 1], [0])

    def parseLog(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrLog, [0], [0])

    def parseLogSoftmax(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrLogSoftmax, [0], [0])

    def parseLoop(self, node):
        self.assert_node_input_output(node, None, None, input_low_bound=2, output_low_bound=1)
        return self.parse_general_node(node, AttrGRU, range(len(node.input)), range(len(node.output)))

    def parseLpNormalization(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrLpNormalization, [0], [0])

    def parseLpPool(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrLpPool, [0], [0])

    def parseMatMul(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrMatMul, [0, 1], [0])

    def parseMatMulInteger(self, node):
        self.assert_node_input_output(node, [2, 3, 4], [1])
        return self.parse_general_node(node, AttrMatMulInteger, range(len(node.input)), [0])

    def parseMax(self, node):
        self.assert_node_input_output(node, None, [1], input_low_bound=1)
        return self.parse_general_node(node, AttrMax, range(len(node.input)), [0])

    def parseMaxPool(self, node):
        self.assert_node_input_output(node, [1], [1, 2])
        return self.parse_general_node(node, AttrMaxPool, [0], range(len(node.output)))

    def parseMaxRoiPool(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrMaxRoiPool, [0, 1], [0])

    def parseMaxUnpool(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        return self.parse_general_node(node, AttrMaxUnpool, range(len(node.input)), [0])

    def parseMean(self, node):
        self.assert_node_input_output(node, None, [1], input_low_bound=1)
        return self.parse_general_node(node, AttrMean, range(len(node.input)), [0])

    def parseMeanVarianceNormalization(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrMeanVarianceNormalization, [0], [0])

    def parseMin(self, node):
        self.assert_node_input_output(node, None, [1], input_low_bound=1)
        return self.parse_general_node(node, AttrMin, range(len(node.input)), [0])

    def parseMod(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrMod, [0, 1], [0])

    def parseMul(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrMul, [0, 1], [0])

    def parseMultinomial(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrMultinomial, [0], [0])

    def parseNeg(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrNeg, [0], [0])

    def parseNegativeLogLikelihoodLoss(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        return self.parse_general_node(node, AttrNegativeLogLikelihoodLoss, range(len(node.input)), [0])

    def parseNonMaxSuppression(self, node):
        self.assert_node_input_output(node, [2, 3, 4, 5], [1])
        return self.parse_general_node(node, AttrNonMaxSuppression, range(len(node.input)), [0])

    def parseNonZero(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrNonZero, [0], [0])

    def parseNot(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrNot, [0], [0])

    def parseOneHot(self, node):
        self.assert_node_input_output(node, [3], [1])
        return self.parse_general_node(node, AttrOneHot, [0, 1, 2], [0])

    def parseOr(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrOr, [0, 1], [0])

    def parsePRelu(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrPRelu, [0, 1], [0])

    def parsePad(self, node):
        self.assert_node_input_output(node, [1, 2, 3], [1])
        return self.parse_general_node(node, AttrPad, range(len(node.input)), [0])

    def parsePow(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrPow, [0, 1], [0])

    def parseQLinearConv(self, node):
        self.assert_node_input_output(node, [8, 9], [1])
        attrs = self.parse_attrs(node.attribute)

        # Get input channel and output channel size
        conn_input_ids = list(range(node.input))
        if node.input[3] in self.input_sizes:
            # param
            output_channel, input_channel = self.input_sizes[node.input[3]][:2]
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = self.input_sizes[node.input[3]][-2:]
            conn_input_ids.remove(3)
        else:
            # weight as node
            output_channel, input_channel = -1, -1
            if 'kernel_shape' not in attrs:
                attrs['kernel_shape'] = -1

        G_attr = AttrQLinearConv(input_channel, output_channel, name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, [0])

    def parseQLinearMatMul(self, node):
        self.assert_node_input_output(node, [8], [1])
        return self.parse_general_node(node, AttrQLinearMatMul, range(len(node.input)), [0])

    def parseQuantizeLinear(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        return self.parse_general_node(node, AttrQuantizeLinear, range(len(node.input)), [0])

    def parseRNN(self, node):
        self.assert_node_input_output(node, [3, 4, 5, 6], [0, 1, 2])
        return self.parse_general_node(node, AttrRNN, range(len(node.input)), range(len(node.output)))

    def parseRandomNormal(self, node):
        self.assert_node_input_output(node, [0], [1])
        return self.parse_general_node(node, AttrRandomNormal, [], [0])

    def parseRandomNormalLike(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrRandomNormalLike, [0], [0])

    def parseRandomUniform(self, node):
        self.assert_node_input_output(node, [0], [1])
        return self.parse_general_node(node, AttrRandomUniform, [], [0])

    def parseRandomUniformLike(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrRandomUniformLike, [0], [0])

    def parseRange(self, node):
        self.assert_node_input_output(node, [3], [1])
        return self.parse_general_node(node, AttrRange, [0, 1, 2], [0])

    def parseReciprocal(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReciprocal, [0], [0])

    def parseReduceL1(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceL1, [0], [0])

    def parseReduceL2(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceL2, [0], [0])

    def parseReduceLogSum(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceLogSum, [0], [0])

    def parseReduceLogSumExp(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceLogSumExp, [0], [0])

    def parseReduceMax(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceMax, [0], [0])

    def parseReduceMean(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceMean, [0], [0])

    def parseReduceMin(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceMin, [0], [0])

    def parseReduceProd(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceProd, [0], [0])

    def parseReduceSum(self, node):
        self.assert_node_input_output(node, [1, 2], [1])
        return self.parse_general_node(node, AttrReduceSum, range(len(node.input)), [0])

    def parseReduceSumSquare(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrReduceSumSquare, [0], [0])

    def parseRelu(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrRelu, [0], [0])

    def parseReshape(self, node):
        self.assert_node_input_output(node, [1, 2], [1])
        attrs = self.parse_attrs(node.attribute)

        # opset = [1, 5), length of node input == 1 and attr has shape
        conn_input_ids = [0]
        if 'shape' in attrs:
            shapes = attrs['shape']
        else:
            if node.input[1] in self.input_sizes:
                # params
                shapes = self.input_sizes[node.input[1]]
            else:
                # shape as node
                shapes = ()
                conn_input_ids.append(1)

        G_attr = AttrReshape(shapes, name = node.output[0], **attrs)
        return self.get_networkx_node_edges(node, G_attr, conn_input_ids, [0])

    def parseResize(self, node):
        self.assert_node_input_output(node, [1, 2, 3, 4], [1])
        return self.parse_general_node(node, AttrResize, range(len(node.input)), [0])

    def parseReverseSequence(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrReverseSequence, [0, 1], [0])

    def parseRoiAlign(self, node):
        # note: official onnx only support 3 input, nart support 2 input
        self.assert_node_input_output(node, [2, 3], [1])
        return self.parse_general_node(node, AttrRoiAlign, range(len(node.input)), [0])

    def parseRound(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrRound, [0], [0])

    def parseScan(self, node):
        self.assert_node_input_output(node, None, None, input_low_bound=1, output_low_bound=1)
        return self.parse_general_node(node, AttrScan, range(len(node.input)), range(len(node.output)))

    def parseScatter(self, node):
        self.assert_node_input_output(node, [3], [1])
        return self.parse_general_node(node, AttrScatter, [0, 1, 2], [0])

    def parseScatterElements(self, node):
        self.assert_node_input_output(node, [3], [1])
        return self.parse_general_node(node, AttrScatterElements, [0, 1, 2], [0])

    def parseScatterND(self, node):
        self.assert_node_input_output(node, [3], [1])
        return self.parse_general_node(node, AttrScatterND, [0, 1, 2], [0])

    def parseSelu(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSelu, [0], [0])

    def parseSequenceAt(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrSequenceAt, [0, 1], [0])

    def parseSequenceConstruct(self, node):
        self.assert_node_input_output(node, None, [1], input_low_bound=1)
        return self.parse_general_node(node, AttrSequenceConstruct, range(len(node.input)), [0])

    def parseSequenceEmpty(self, node):
        self.assert_node_input_output(node, [0], [1])
        return self.parse_general_node(node, AttrSequenceEmpty, [], [0])

    def parseSequenceErase(self, node):
        self.assert_node_input_output(node, [1, 2], [1])
        return self.parse_general_node(node, AttrSequenceErase, range(len(node.input)), [0])

    def parseSequenceInsert(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        return self.parse_general_node(node, AttrSequenceInsert, range(len(node.input)), [0])

    def parseSequenceLength(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSequenceLength, [0], [0])

    def parseShape(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrShape, [0], [0])

    def parseShrink(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrShrink, [0], [0])

    def parseSigmoid(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSigmoid, [0], [0])

    def parseSign(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSign, [0], [0])

    def parseSin(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSin, [0], [0])

    def parseSinh(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSinh, [0], [0])

    def parseSize(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSize, [0], [0])

    def parseSlice(self, node):
        self.assert_node_input_output(node, [1, 3, 4, 5], [1])
        return self.parse_general_node(node, AttrSlice, range(len(node.input)), [0])

    def parseSoftmax(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSoftmax, [0], [0])

    def parseSoftmaxCrossEntropyLoss(self, node):
        self.assert_node_input_output(node, [2, 3], [1, 2])
        return self.parse_general_node(node, AttrSoftmaxCrossEntropyLoss, range(len(node.input)), range(len(node.output)))

    def parseSoftplus(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSoftplus, [0], [0])

    def parseSoftsign(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSoftsign, [0], [0])

    def parseSpaceToDepth(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSpaceToDepth, [0], [0])

    def parseSplit(self, node):
        self.assert_node_input_output(node, [1, 2], None, output_low_bound=1)
        return self.parse_general_node(node, AttrSplit, range(len(node.input)), range(len(node.output)))

    def parseSplitToSequence(self, node):
        self.assert_node_input_output(node, [1, 2], [1])
        return self.parse_general_node(node, AttrSplitToSequence, range(len(node.input)), [0])

    def parseSqrt(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrSqrt, [0], [0])

    def parseSqueeze(self, node):
        self.assert_node_input_output(node, [1, 2], [1])
        return self.parse_general_node(node, AttrSqueeze, range(len(node.input)), [0])

    def parseStringNormalizer(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrStringNormalizer, [0], [0])

    def parseSub(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrSub, [0, 1], [0])

    def parseSum(self, node):
        self.assert_node_input_output(node, None, [1], input_low_bound=1)
        return self.parse_general_node(node, AttrSum, range(len(node.input)), [0])

    def parseTan(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrTan, [0], [0])

    def parseTanh(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrTanh, [0], [0])

    def parseTfIdfVectorizer(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrTfIdfVectorizer, [0], [0])

    def parseThresholdedRelu(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrThresholdedRelu, [0], [0])

    def parseTile(self, node):
        self.assert_node_input_output(node, [2, 3], [1])
        return self.parse_general_node(node, AttrTile, range(len(node.input)), [0])

    def parseTopK(self, node):
        self.assert_node_input_output(node, [1, 2], [2])
        return self.parse_general_node(node, AttrTopK, range(len(node.input)), [0, 1])

    def parseTranspose(self, node):
        self.assert_node_input_output(node, [1], [1])
        return self.parse_general_node(node, AttrTranspose, [0], [0])

    def parseUnique(self, node):
        self.assert_node_input_output(node, [1], [1, 2, 3, 4])
        return self.parse_general_node(node, AttrUnique, [0], range(len(node.output)))

    def parseUnsqueeze(self, node):
        self.assert_node_input_output(node, [1, 2], [1])
        return self.parse_general_node(node, AttrUnsqueeze, range(len(node.input)), [0])

    def parseUpsample(self, node):
        self.assert_node_input_output(node, [1, 2], [1])
        return self.parse_general_node(node, AttrUpsample, range(len(node.input)), [0])

    def parseWhere(self, node):
        self.assert_node_input_output(node, [3], [1])
        return self.parse_general_node(node, AttrWhere, [0, 1, 2], [0])

    def parseXor(self, node):
        self.assert_node_input_output(node, [2], [1])
        return self.parse_general_node(node, AttrWhere, [0, 1], [0])