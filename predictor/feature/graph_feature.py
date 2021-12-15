import numpy as np
import networkx as nx
from .onnx_to_networkx import onnx2nx
from .onnx_shape_infer import custom_shape_infer
from .onnx_flops import calculate_onnx_flops
from .node_feature import extract_node_features


def modify_onnx_batch_size(onnx_G, batch_size):
    # initializer names, in case of names of input include initializers
    init_names = set()
    for init in onnx_G.graph.initializer:
        init_names.add(init.name)

    def _modify(node, dim_idx, value):
        dims = node.type.tensor_type.shape.dim
        if len(dims) > dim_idx:
            value = value[node.name] if isinstance(value, dict) else value
            dims[dim_idx].dim_value = value

    # modify input
    for inp in onnx_G.graph.input:
        if inp.name in init_names:
            continue
        _modify(inp, 0, batch_size)

    # modify output
    for out in onnx_G.graph.output:
        _modify(out, 0, batch_size)

    return


def parse_from_onnx(onnx_path, batch_size):
    pG = onnx2nx(onnx_path)
    nx_G, onnx_G = pG.data, pG.onnx_G

    # first we should change the batch_size of input in ONNX model
    modify_onnx_batch_size(onnx_G, batch_size)
    status, newG, output_shapes = custom_shape_infer(onnx_G)
    # if failed modify the batch to original batch size
    assert status is True, "Onnx shape infer error!"

    flops, params, macs, node_flops = calculate_onnx_flops(onnx_G, True)
    return nx_G, output_shapes, flops, params, macs, node_flops, newG


def extract_graph_feature_from_networkx(nx_G, batch_size, output_shapes, flops, params, macs, undirected=True):
    # static features: flops, params, memory_access (GB) + batch_size
    static_features = np.array([batch_size, flops / 1e9, params / 1e9, macs / 1e9], dtype="float32")

    # node features
    node_features = extract_node_features(nx_G, output_shapes, batch_size)

    # get features conducted by idx
    features = []
    name2id = {}
    id2name = {}
    for idx, node in enumerate(nx.topological_sort(nx_G)):
        features.append(node_features[node])
        name2id[node] = idx
        id2name[idx] = node

    # get graph adjacent matrix
    node_num = nx_G.number_of_nodes()
    adjacent = np.zeros((node_num, node_num), dtype="float32")

    for node in nx_G.nodes():
        idx = name2id[node]
        for child in nx_G.successors(node):
            conn_idx = name2id[child]
            adjacent[idx][conn_idx] = 1
            if undirected:
                adjacent[conn_idx][idx] = 1

    # test connect relationship
    # xs, ys = np.where(adjacent > 0)
    # for i in range(len(xs)):
    #     print("Conn:", id2name[xs[i]], id2name[ys[i]])

    # feature in features may be a tuple (block_adjacent, block_features, block_static_features)
    return adjacent, features, static_features


def extract_graph_feature(onnx_path, batch_size, return_onnx=False):
    nx_G, output_shapes, flops, params, macs, node_flops, onnx_G = parse_from_onnx(onnx_path, batch_size)
    adjacent, features, static_features = extract_graph_feature_from_networkx(
        nx_G, batch_size, output_shapes, flops, params, macs
    )

    if return_onnx:
        return adjacent, np.array(features), static_features, onnx_G
    else:
        return adjacent, np.array(features), static_features
