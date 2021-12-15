import onnx
from onnx.shape_inference import infer_shapes


def onnx_node_attributes_to_dict(args):
    def onnx_attribute_to_dict(onnx_attr):
        if onnx_attr.HasField('t'):
            return onnx.numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def get_1D_const_node(name, data_type, vals):
    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=onnx.helper.make_tensor(
            name=name + '_value',
            data_type=data_type,
            dims=(len(vals), ),
            vals=vals,
        )
    )
    return node


def custom_shape_infer(onnx_G):

    vals = [x for x in onnx_G.graph.value_info]
    for x in vals:
        onnx_G.graph.value_info.remove(x)

    def parse_dim(val):
        shape = list([x.dim_value for x in val.type.tensor_type.shape.dim])
        shape = shape if len(shape) > 0 else None
        return shape

    output_shapes = {}
    init_dtypes = {}
    placehoders = []

    # parse initializer
    for init in onnx_G.graph.initializer:
        output_shapes[init.name] = list(init.dims)
        init_dtypes[init.name] = init.data_type

    # parse input
    for val in onnx_G.graph.input:
        output_shapes[val.name] = parse_dim(val)
        dtype = val.type.tensor_type.elem_type

        # init dtype may != input dtype, correct it in case of shape inference error
        if val.name in init_dtypes and dtype != init_dtypes[val.name]:
            print(" - Warning: {}: init type:{} vs input type:{}".format(val.name, init_dtypes[val.name], dtype))
            val.type.tensor_type.elem_type = init_dtypes[val.name]

        # placehoders for the whole graph
        if val.name not in init_dtypes:
            placehoders.append(val)

    # parse output
    output_vals = {}
    for val in onnx_G.graph.output:
        output_shapes[val.name] = parse_dim(val)
        output_vals[val.name] = val

    # parse node, handle error conditions
    vers = ['1.7.0', '1.8.1']
    if onnx.__version__ not in vers:
        print(" - Warning: onnx version should in {}, but with {}, infer shape may cause error" \
            .format(vers, onnx.__version__))

    # infer the shapes by onnx api
    if onnx.__version__ == '1.8.1':
        G = infer_shapes(onnx_G, strict_mode=True)
    # args strict_mode is not introcuded for onnx == 1.7.0
    else:
        G = infer_shapes(onnx_G)

    for val in G.graph.value_info:
        output_shapes[val.name] = parse_dim(val)

    miss_ops = {}
    for node in G.graph.node:

        # constant output could be lost when it is =[]
        if node.op_type == "Constant":
            out = node.output[0]
            if out not in output_shapes or output_shapes[out] is None:
                for attr in node.attribute:
                    if attr.name == "value":
                        output_shapes[out] = list(attr.t.dims)

        # check if the shapes inferred are complete
        for output in node.output:
            if output not in output_shapes or output_shapes[output] is None:
                if node.op_type not in miss_ops:
                    miss_ops[node.op_type] = (node.name, node.output[0])

    for k, v in miss_ops.items():
        print(" - Warning: Miss shape infer, op=[{}], name=[{}], output=[{}]!".format(k, v[0], v[1]))
    return False if miss_ops else True, G, output_shapes


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    onnx_G = onnx.load(path)
    status, newG, output_shapes = custom_shape_infer(onnx_G)
    print(status)
