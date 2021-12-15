# decorator for defining attributes in different onnx opsets
def ATTR(*out_args, **out_kwargs):
    def wrapper1(func):
        def wrapper2(*in_args, **in_kwargs):

            # check if current opset support current attribute
            if check_version(in_kwargs['opsets'], out_kwargs['opsets']):
                attr_name = out_kwargs['name']

                if attr_name not in in_kwargs:
                    if 'default' not in out_kwargs:
                        # opsets = [9, 10] for Slice, starts and ends
                        # will be required for opset 9 but deprecated for opset 10.
                        # the check will cause error
                        raise Exception("ATTR {} <opsets={}> is required while constructing!" \
                            .format(attr_name, in_kwargs['opsets']))
                    attr_value = out_kwargs['default']
                else:
                    attr_value = in_kwargs[attr_name]
                    del in_kwargs[attr_name]

                # format functions, such like dilations, kernel_shape, ...
                if 'fmt_func' in out_kwargs:
                    attr_value = out_kwargs['fmt_func'](attr_value)
                in_args[0].attributes[attr_name] = attr_value

            f = func(*in_args, **in_kwargs)
            return f
        return wrapper2
    return wrapper1


# check if attr opset versions contain target opset versions
def check_version(target_opsets, attr_opset):
    for v in target_opsets:
        if len(attr_opset) == 1:
            if v >= attr_opset[0]:
                return True
        elif len(attr_opset) == 2:
            if v >= attr_opset[0] and v < attr_opset[1]:
                return True
        else:
            raise Exception("ATTR opset length should be 1 or 2!")
    return False


WARNINGS = set()
def check_redundant_attrs(attrs, class_name=""):
    global WARNINGS
    tv = attrs['opsets']
    del attrs['opsets']
    if (not attrs) is False:
        WARNINGS.add("Warnings: unused args for <{}> in opsets <{}>: {}!".format(class_name, tv, attrs))


class FSize:

    @staticmethod
    def kernel(x):
        return FSize.tuple2(x, "Kernel")

    @staticmethod
    def stride(x):
        return FSize.tuple2(x, "Stride")

    @staticmethod
    def padding(x):
        return FSize.tuple4(x, "Padding")

    @staticmethod
    def dilation(x):
        return FSize.tuple2(x, "Dilation")

    @staticmethod
    def tuple2(x, note):
        if isinstance(x, int):
            return (x, x)
        if isinstance(x, list):
            x = tuple(x)
        if isinstance(x, tuple) and len(x) == 2:
            return x
        raise Exception("Unexcepted {} size: {}!".format(note, x))

    @staticmethod
    def tuple4(x, note):
        if isinstance(x, int):
            return (x, x, x, x)
        if isinstance(x, list):
            x = tuple(x)
        if isinstance(x, tuple) and len(x) == 2:
            return (x[0], x[0], x[1], x[1])
        if isinstance(x, tuple) and len(x) == 4:
            return x
        raise Exception("Unexcepted {} size: {}!".format(note, x))


# base class for attributes of oerations
class AttrBaseOp(object):
    def __getattr__(self, k):

        # attribtes that is not important
        # different values in self.attributes
        if k == "node_properties":
            setattr(self, "node_properties", {})
            return getattr(self, "node_properties")

        # attributes to be hashed and compared
        # different values in self.attributes denotes different nodes
        elif k == "attributes":
            setattr(self, "attributes", {})
            return getattr(self, "attributes")

        else:
            raise(AttributeError(k))

# -----------------------------------------------------------------------------
#                   Attribute Class For Specific Operator
# -----------------------------------------------------------------------------

# This file defines attributes of available operators from onnx
class AttrInput(AttrBaseOp):

    def __init__(self, name=''):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Input'


class AttrAbs(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Abs'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrAcos(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Acos'


class AttrAcosh(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Acosh'


class AttrAdd(AttrBaseOp):

    @ATTR(name='axis',            opsets=(1, 7), default=0)  # v1 add, v7 delete
    @ATTR(name='broadcast',       opsets=(1, 7), default=0)  # v1 add, v7 delete
    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Add'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrAnd(AttrBaseOp):

    @ATTR(name='axis',      opsets=(1, 7), default=0) # v1 add, v7 delete
    @ATTR(name='broadcast', opsets=(1, 7), default=0) # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'And'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrArgMax(AttrBaseOp):

    @ATTR(name='axis',              opsets=( 1, ), default=0) # v1 add
    @ATTR(name='keepdims',          opsets=( 1, ), default=1) # v1 add
    @ATTR(name='select_last_index', opsets=(12, ), default=0) # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ArgMax'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrArgMin(AttrBaseOp):

    @ATTR(name='axis',              opsets=( 1, ), default=0) # v1 add
    @ATTR(name='keepdims',          opsets=( 1, ), default=1) # v1 add
    @ATTR(name='select_last_index', opsets=(12, ), default=0) # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ArgMin'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrAsin(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Asin'


class AttrAsinh(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Asinh'


class AttrAtan(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Atan'


class AttrAtanh(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Atanh'


class AttrAveragePool(AttrBaseOp):

    @ATTR(name='auto_pad',          opsets=( 1, ), default='NOTSET')                  # v1 add
    @ATTR(name='ceil_mode',         opsets=(10, ), default=0)                         # v10 add
    @ATTR(name='count_include_pad', opsets=( 7, ), default=0)                         # v7 add
    @ATTR(name='pads',              opsets=( 1, ), default=0, fmt_func=FSize.padding) # v1 add
    @ATTR(name='strides',           opsets=( 1, ), default=1, fmt_func=FSize.stride)  # v1 add
    @ATTR(name='kernel_shape',      opsets=( 1, ),            fmt_func=FSize.kernel)  # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'AveragePool'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrBatchNormalization(AttrBaseOp):

    @ATTR(name='epsilon',         opsets=(1,  ), default=1e-5) # v1 add
    @ATTR(name='momentum',        opsets=(1,  ), default=0.9)  # v1 add
    @ATTR(name='consumed_inputs', opsets=(1, 6))               # v1 add, v6 delete, required
    @ATTR(name='is_test',         opsets=(1, 7), default=0)    # v1 add, v7 delete
    @ATTR(name='spatial',         opsets=(1, 9), default=1)    # v1 add, v9 delete
    def __init__(self, scale_size, B_size, mean_size, var_size, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'BatchNormalization'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # sizes of scale, B, mean, var from weights
        self.attributes['scale_size'] = scale_size
        self.attributes['B_size'] = B_size
        self.attributes['mean_size'] = mean_size
        self.attributes['var_size'] = var_size


class AttrBitShift(AttrBaseOp):

    @ATTR(name='direction', opsets=(11, )) # v11 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'BitShift'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrCast(AttrBaseOp):

    @ATTR(name='to', opsets=(1, )) # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Cast'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrCeil(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Ceil'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrCelu(AttrBaseOp):

    @ATTR(name='alpha', opsets=(12, ), default=1.0) # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Celu'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrClip(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1,  6), default=())            # v1 add
    @ATTR(name='max',             opsets=(1, 11), default=3.402823e+38)  # v1 add, v11 delete
    @ATTR(name='min',             opsets=(1, 11), default=-3.402823e+38) # v1 add, v11 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Clip'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrCompress(AttrBaseOp):

    @ATTR(name='axis', opsets=(9, ), default=0)            # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Compress'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrConcat(AttrBaseOp):

    @ATTR(name='axis', opsets=(1, 4), default=1) # v1 add, v4 delete
    @ATTR(name='axis', opsets=(4,  ))            # v4 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Concat'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrConcatFromSequence(AttrBaseOp):

    @ATTR(name='axis',     opsets=(11, ))            # v11 add, required
    @ATTR(name='new_axis', opsets=(11, ), default=0) # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ConcatFromSequence'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrConstant(AttrBaseOp):

    @ATTR(name='value',         opsets=( 1, ), default=None) # v1 add,  record dims not value
    @ATTR(name='sparse_value',  opsets=(11, ), default=None) # v11 add, record dims not value
    @ATTR(name='value_float',   opsets=(12, ), default=0.0)  # v12 add
    @ATTR(name='value_floats',  opsets=(12, ), default=())   # v12 add
    @ATTR(name='value_int',     opsets=(12, ), default=0)    # v12 add
    @ATTR(name='value_ints',    opsets=(12, ), default=())   # v12 add
    @ATTR(name='value_string',  opsets=(12, ), default='')   # v12 add
    @ATTR(name='value_strings', opsets=(12, ), default=())   # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Constant'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrConstantOfShape(AttrBaseOp):

    @ATTR(name='value',             opsets=(9, ), default=None) # v9 add, record values
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ConstantOfShape'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrConv(AttrBaseOp):

    @ATTR(name='auto_pad',     opsets=(1, ), default='NOTSET')                   # v1 add
    @ATTR(name='group',        opsets=(1, ), default=1)                          # v1 add
    @ATTR(name='pads',         opsets=(1, ), default=0, fmt_func=FSize.padding)  # v1 add
    @ATTR(name='strides',      opsets=(1, ), default=1, fmt_func=FSize.stride)   # v1 add
    @ATTR(name='dilations',    opsets=(1, ), default=1, fmt_func=FSize.dilation) # v1 add
    @ATTR(name='kernel_shape', opsets=(1, ),            fmt_func=FSize.kernel)   # v1 add, required
    def __init__(self, input_channel, output_channel, bias, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Conv'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # input_channel & output_channel from weights
        self.attributes['input_channel'] = input_channel
        self.attributes['output_channel'] = output_channel
        self.attributes['bias'] = bias


class AttrConvInteger(AttrBaseOp):

    @ATTR(name='auto_pad',     opsets=(10, ), default='NOTSET')                   # v10 add
    @ATTR(name='group',        opsets=(10, ), default=1)                          # v10 add
    @ATTR(name='pads',         opsets=(10, ), default=0, fmt_func=FSize.padding)  # v10 add
    @ATTR(name='strides',      opsets=(10, ), default=1, fmt_func=FSize.stride)   # v10 add
    @ATTR(name='dilations',    opsets=(10, ), default=1, fmt_func=FSize.dilation) # v10 add
    @ATTR(name='kernel_shape', opsets=(10, ),            fmt_func=FSize.kernel)   # v10 add, required
    def __init__(self, input_channel, output_channel, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ConvInteger'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # input_channel & output_channel from weights
        self.attributes['input_channel'] = input_channel
        self.attributes['output_channel'] = output_channel


class AttrConvTranspose(AttrBaseOp):

    @ATTR(name='auto_pad',       opsets=(1, ), default='NOTSET')                   # v1 add
    @ATTR(name='group',          opsets=(1, ), default=1)                          # v1 add
    @ATTR(name='output_shape',   opsets=(1, ), default=())                         # v1 add
    @ATTR(name='pads',           opsets=(1, ), default=0, fmt_func=FSize.padding)  # v1 add
    @ATTR(name='strides',        opsets=(1, ), default=1, fmt_func=FSize.stride)   # v1 add
    @ATTR(name='dilations',      opsets=(1, ), default=1, fmt_func=FSize.dilation) # v1 add
    @ATTR(name='output_padding', opsets=(1, ), default=0, fmt_func=FSize.padding)  # v1 add
    @ATTR(name='kernel_shape',   opsets=(1, ),            fmt_func=FSize.kernel)   # v1 add, required
    def __init__(self, input_channel, output_channel, bias, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ConvTranspose'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # input_channel & output_channel from weights
        self.attributes['input_channel'] = input_channel
        self.attributes['output_channel'] = output_channel
        self.attributes['bias'] = bias


class AttrCos(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Cos'


class AttrCosh(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Cosh'


class AttrCumSum(AttrBaseOp):

    @ATTR(name='exclusive', opsets=(11, ), default=0) # v11 add
    @ATTR(name='reverse',   opsets=(11, ), default=0) # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'CumSum'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrDepthToSpace(AttrBaseOp):

    @ATTR(name='blocksize', opsets=( 1, ))                # v1 add, required
    @ATTR(name='mode',      opsets=(11, ), default='DCR') # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'DepthToSpace'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrDequantizeLinear(AttrBaseOp):

    @ATTR(name='axis', opsets=(13, ), default=1) # v13 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'DequantizeLinear'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrDet(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Det'


class AttrDiv(AttrBaseOp):

    @ATTR(name='axis',            opsets=(1, 7), default=0)  # v1 add, v7 delete
    @ATTR(name='broadcast',       opsets=(1, 7), default=0)  # v1 add, v7 delete
    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Div'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrDropout(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=( 1,  6), default=())  # v1 add, v6 delete
    @ATTR(name='is_test',         opsets=( 1,  7), default=0)   # v1 add, v7 delete
    @ATTR(name='ratio',           opsets=( 1, 12), default=0.5) # v1 add, v12 delete
    @ATTR(name='seed',            opsets=(12,   ), default=0)   # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Dropout'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrDynamicQuantizeLinear(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'DynamicQuantizeLinear'


class AttrEinsum(AttrBaseOp):

    @ATTR(name='equation', opsets=(12, )) # v12 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Einsum'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrElu(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=())  # v1 add, v6 delete
    @ATTR(name='alpha',           opsets=(1,  ), default=1.0) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Elu'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrEqual(AttrBaseOp):

    @ATTR(name='axis',      opsets=(1, 7), default=0) # v1 add, v7 delete
    @ATTR(name='broadcast', opsets=(1, 7), default=0) # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Equal'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrErf(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Erf'


class AttrExp(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Exp'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrExpand(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Expand'


class AttrEyeLike(AttrBaseOp):

    @ATTR(name='dtype', opsets=(9, ), default=0)  # v9 add
    @ATTR(name='k',     opsets=(9, ), default=0)  # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'EyeLike'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrFlatten(AttrBaseOp):

    @ATTR(name='axis', opsets=(1, ), default=1) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Flatten'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrFloor(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Floor'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrGRU(AttrBaseOp):

    @ATTR(name='activation_alpha',    opsets=(1,  ), default=())        # v1 add
    @ATTR(name='activation_beta',     opsets=(1,  ), default=())        # v1 add
    @ATTR(name='activations',         opsets=(1,  ), default=())        # v1 add
    @ATTR(name='clip',                opsets=(1,  ), default=0)         # v1 add
    @ATTR(name='direction',           opsets=(1,  ), default='forward') # v1 add
    @ATTR(name='hidden_size',         opsets=(1,  ), default=1)         # v1 add
    @ATTR(name='output_sequence',     opsets=(1, 7), default=0)         # v1 add, v7 delete
    @ATTR(name='linear_before_reset', opsets=(3,  ), default=0)         # v3 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'GRU'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrGather(AttrBaseOp):

    @ATTR(name='axis', opsets=(1,  ), default=0) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Gather'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrGatherElements(AttrBaseOp):

    @ATTR(name='axis', opsets=(11,  ), default=0) # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'GatherElements'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrGatherND(AttrBaseOp):

    @ATTR(name='batch_dims', opsets=(12,  ), default=0) # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'GatherND'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrGemm(AttrBaseOp):

    @ATTR(name='alpha',     opsets=(1,  ), default=1.0) # v1 add
    @ATTR(name='beta',      opsets=(1,  ), default=1.0) # v1 add
    @ATTR(name='broadcast', opsets=(1, 7), default=0)   # v1 add, v7 delete
    @ATTR(name='transA',    opsets=(1,  ), default=0)   # v1 add
    @ATTR(name='transB',    opsets=(1,  ), default=0)   # v1 add
    def __init__(self, input_size, output_size, bias, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Gemm'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # input_size & output_size from weights
        self.attributes['input_size'] = input_size
        self.attributes['output_size'] = output_size
        self.attributes['bias'] = bias


class AttrGlobalAveragePool(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'GlobalAveragePool'


class AttrGlobalLpPool(AttrBaseOp):

    @ATTR(name='p', opsets=(1, ), default=2) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'GlobalLpPool'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrGlobalMaxPool(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'GlobalMaxPool'


class AttrGreater(AttrBaseOp):

    @ATTR(name='axis',      opsets=(1, 7), default=0) # v1 add, v7 delete
    @ATTR(name='broadcast', opsets=(1, 7), default=0) # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Greater'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrGreaterOrEqual(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'GreaterOrEqual'


class AttrHardSigmoid(AttrBaseOp):

    @ATTR(name='alpha',           opsets=(1,  ), default=0.2) # v1 add
    @ATTR(name='beta',            opsets=(1,  ), default=0.5) # v1 add
    @ATTR(name='consumed_inputs', opsets=(1, 6), default=())  # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'HardSigmoid'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrHardmax(AttrBaseOp):

    @ATTR(name='axis', opsets=( 1, 13), default=1)  # v1 add, v13 delete
    @ATTR(name='axis', opsets=(13,   ), default=-1) # v13 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Hardmax'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrIdentity(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Identity'


class AttrIf(AttrBaseOp):

    @ATTR(name='else_branch', opsets=(1, )) # v1 add
    @ATTR(name='then_branch', opsets=(1, )) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'If'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrInstanceNormalization(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=())    # v1 add, v6 delete
    @ATTR(name='epsilon',         opsets=(1,  ), default=1e-05) # v1 add
    def __init__(self, scale_size, B_size, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'InstanceNormalization'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # sizes of scale, B from weights
        self.attributes['scale_size'] = scale_size
        self.attributes['B_size'] = B_size


class AttrIsInf(AttrBaseOp):

    @ATTR(name='detect_negative', opsets=(10, ), default=1) # v10 add
    @ATTR(name='detect_positive', opsets=(10, ), default=1) # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'IsInf'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrIsNaN(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'IsNaN'


class AttrLRN(AttrBaseOp):

    @ATTR(name='alpha', opsets=(1, ), default=0.0001) # v1 add
    @ATTR(name='beta',  opsets=(1, ), default=0.75)   # v1 add
    @ATTR(name='bias',  opsets=(1, ), default=1.0)    # v1 add
    @ATTR(name='size',  opsets=(1, ))                 # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'LRN'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLSTM(AttrBaseOp):

    @ATTR(name='activation_alpha',    opsets=(1,  ), default=())        # v1 add
    @ATTR(name='activation_beta',     opsets=(1,  ), default=())        # v1 add
    @ATTR(name='activations',         opsets=(1,  ), default=())        # v1 add
    @ATTR(name='clip',                opsets=(1,  ), default=0)         # v1 add
    @ATTR(name='direction',           opsets=(1,  ), default='forward') # v1 add
    @ATTR(name='hidden_size',         opsets=(1,  ), default=1)         # v1 add
    @ATTR(name='input_forget',        opsets=(1,  ), default=0)         # v1 add
    @ATTR(name='output_sequence',     opsets=(1, 7), default=0)         # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'LSTM'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLeakyRelu(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=())   # v1 add, v6 delete
    @ATTR(name='alpha',           opsets=(1,  ), default=0.01) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'LeakyRelu'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLess(AttrBaseOp):

    @ATTR(name='axis',      opsets=(1, 7), default=0) # v1 add, v7 delete
    @ATTR(name='broadcast', opsets=(1, 7), default=0) # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Less'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLessOrEqual(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'LessOrEqual'


class AttrLog(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Log'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLogSoftmax(AttrBaseOp):

    @ATTR(name='axis', opsets=( 1, 13), default=1)  # v1 add, v13 delete
    @ATTR(name='axis', opsets=(13,   ), default=-1) # v13 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'LogSoftmax'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLoop(AttrBaseOp):

    @ATTR(name='body', opsets=(1, )) # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Loop'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLpNormalization(AttrBaseOp):

    @ATTR(name='axis', opsets=(1, ), default=-1) # v1 add
    @ATTR(name='p',    opsets=(1, ), default=2)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'LpNormalization'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrLpPool(AttrBaseOp):

    @ATTR(name='auto_pad',      opsets=(1, ), default='NOTSET')                   # v1 add
    @ATTR(name='kernel_shape',  opsets=(1, ),            fmt_func=FSize.kernel)   # v1 add, required
    @ATTR(name='p',             opsets=(1, ), default=2.0)                        # v1 add
    @ATTR(name='pads',          opsets=(1, ), default=0, fmt_func=FSize.padding)  # v1 add
    @ATTR(name='strides',       opsets=(1, ), default=1, fmt_func=FSize.stride)   # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'LpPool'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMatMul(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'MatMul'


class AttrMatMulInteger(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'MatMulInteger'


class AttrMax(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Max'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMaxPool(AttrBaseOp):

    @ATTR(name='auto_pad',      opsets=( 1, ), default='NOTSET')                   # v1 add
    @ATTR(name='kernel_shape',  opsets=( 1, ),            fmt_func=FSize.kernel)   # v1 add, required
    @ATTR(name='pads',          opsets=( 1, ), default=0, fmt_func=FSize.padding)  # v1 add
    @ATTR(name='strides',       opsets=( 1, ), default=1, fmt_func=FSize.stride)   # v1 add
    @ATTR(name='storage_order', opsets=( 8, ), default=0)                          # v8 add
    @ATTR(name='ceil_mode',     opsets=(10, ), default=0)                          # v10 add
    @ATTR(name='dilations',     opsets=(10, ), default=1, fmt_func=FSize.dilation) # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'MaxPool'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMaxRoiPool(AttrBaseOp):

    @ATTR(name='pooled_shape',  opsets=(1, ))              # v1 add
    @ATTR(name='spatial_scale', opsets=(1, ), default=1.0) # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'MaxRoiPool'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMaxUnpool(AttrBaseOp):

    @ATTR(name='kernel_shape', opsets=(9, ))            # v9 add
    @ATTR(name='pads',         opsets=(9, ), default=0) # v9 add
    @ATTR(name='strides',      opsets=(9, ), default=1) # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'MaxUnpool'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMean(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Mean'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMeanVarianceNormalization(AttrBaseOp):

    @ATTR(name='axes', opsets=(9, ), default=[0, 2, 3]) # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'MeanVarianceNormalization'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMin(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Min'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMod(AttrBaseOp):

    @ATTR(name='fmod', opsets=(10, ), default=0) # v10 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Mod'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMul(AttrBaseOp):

    @ATTR(name='axis',              opsets=(1, 7))             # v1 add, v7 delete, required
    @ATTR(name='broadcast',         opsets=(1, 7), default=0)  # v1 add, v7 delete
    @ATTR(name='consumed_inputs',   opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Mul'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrMultinomial(AttrBaseOp):

    @ATTR(name='dtype',       opsets=(7, ), default=6)   # v7 add
    @ATTR(name='sample_size', opsets=(7, ), default=1)   # v7 add
    @ATTR(name='seed',        opsets=(7, ), default=0.0) # v7 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Multinomial'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrNeg(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Neg'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrNegativeLogLikelihoodLoss(AttrBaseOp):

    @ATTR(name='ignore_index', opsets=(12, ), default=-1)     # v12 add
    @ATTR(name='reduction',    opsets=(12, ), default='mean') # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'NegativeLogLikelihoodLoss'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrNonMaxSuppression(AttrBaseOp):

    @ATTR(name='center_point_box', opsets=(10, ), default=0) # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'NonMaxSuppression'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrNonZero(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'NonZero'


class AttrNot(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Not'


class AttrOneHot(AttrBaseOp):

    @ATTR(name='axis', opsets=(9, ), default=-1) # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'OneHot'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrOr(AttrBaseOp):

    @ATTR(name='axis',      opsets=(1, 7), default=0) # v1 add, v7 delete
    @ATTR(name='broadcast', opsets=(1, 7), default=0) # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Or'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrPRelu(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'PRelu'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrPad(AttrBaseOp):

    @ATTR(name='mode',     opsets=(1,   ), default='constant')     # v1 add
    @ATTR(name='paddings', opsets=(1,  2), fmt_func=FSize.padding) # v1 add, required
    @ATTR(name='pads',     opsets=(2, 11), fmt_func=FSize.padding) # v1 add, required
    @ATTR(name='value',    opsets=(1, 11), default=0.0)            # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Pad'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrPow(AttrBaseOp):

    @ATTR(name='axis',      opsets=(1, 7), default=0) # v1 add, v7 delete
    @ATTR(name='broadcast', opsets=(1, 7), default=0) # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Pow'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrQLinearConv(AttrBaseOp):

    @ATTR(name='auto_pad',     opsets=(10, ), default='NOTSET')                   # v10 add
    @ATTR(name='dilations',    opsets=(10, ), default=1, fmt_func=FSize.dilation) # v10 add
    @ATTR(name='group',        opsets=(10, ), default=1)                          # v10 add
    @ATTR(name='kernel_shape', opsets=(10, ),            fmt_func=FSize.kernel)   # v10 add, required
    @ATTR(name='pads',         opsets=(10, ), default=0, fmt_func=FSize.padding)  # v10 add
    @ATTR(name='strides',      opsets=(10, ), default=1, fmt_func=FSize.stride)   # v10 add
    def __init__(self, input_channel, output_channel, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'QLinearConv'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # input_channel & output_channel from weights
        self.attributes['input_channel'] = input_channel
        self.attributes['output_channel'] = output_channel


class AttrQLinearMatMul(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'QLinearMatMul'


class AttrQuantizeLinear(AttrBaseOp):

    @ATTR(name='axis', opsets=(13, ), default=1) # v13 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'QuantizeLinear'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRNN(AttrBaseOp):

    @ATTR(name='activation_alpha',    opsets=(1,  ), default=())        # v1 add
    @ATTR(name='activation_beta',     opsets=(1,  ), default=())        # v1 add
    @ATTR(name='activations',         opsets=(1,  ), default=())        # v1 add
    @ATTR(name='clip',                opsets=(1,  ), default=0)         # v1 add
    @ATTR(name='direction',           opsets=(1,  ), default='forward') # v1 add
    @ATTR(name='hidden_size',         opsets=(1,  ), default=1)         # v1 add
    @ATTR(name='output_sequence',     opsets=(1, 7), default=0)         # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'RNN'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRandomNormal(AttrBaseOp):

    @ATTR(name='dtype', opsets=(1, ), default=1)   # v1 add
    @ATTR(name='mean',  opsets=(1, ), default=0.0) # v1 add
    @ATTR(name='scale', opsets=(1, ), default=1.0) # v1 add
    @ATTR(name='seed',  opsets=(1, ), default=1.0) # v1 add
    @ATTR(name='shape', opsets=(1, ))              # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'RandomNormal'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRandomNormalLike(AttrBaseOp):

    @ATTR(name='dtype', opsets=(1, ), default=-1)  # v1 add
    @ATTR(name='mean',  opsets=(1, ), default=0.0) # v1 add
    @ATTR(name='scale', opsets=(1, ), default=1.0) # v1 add
    @ATTR(name='seed',  opsets=(1, ), default=1.0) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'RandomNormalLike'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRandomUniform(AttrBaseOp):

    @ATTR(name='dtype', opsets=(1, ), default=1)   # v1 add
    @ATTR(name='high',  opsets=(1, ), default=1.0) # v1 add
    @ATTR(name='low',   opsets=(1, ), default=0.0) # v1 add
    @ATTR(name='seed',  opsets=(1, ), default=1.0) # v1 add
    @ATTR(name='shape', opsets=(1, ))              # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'RandomUniform'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRandomUniformLike(AttrBaseOp):

    @ATTR(name='dtype', opsets=(1, ), default=-1)  # v1 add
    @ATTR(name='high',  opsets=(1, ), default=1.0) # v1 add
    @ATTR(name='low',   opsets=(1, ), default=0.0) # v1 add
    @ATTR(name='seed',  opsets=(1, ), default=1.0) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'RandomUniformLike'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRange(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Range'


class AttrReciprocal(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Reciprocal'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceL1(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceL1'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceL2(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceL2'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceLogSum(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceLogSum'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceLogSumExp(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceLogSumExp'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceMax(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceMax'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceMean(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceMean'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceMin(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceMin'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceProd(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceProd'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceSum(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, 13), default=())             # v1 add, v13 delete
    @ATTR(name='keepdims', opsets=(1,   ), default=1)              # v1 add
    @ATTR(name='noop_with_empty_axes', opsets=(13,   ), default=0) # v13 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceSum'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReduceSumSquare(AttrBaseOp):

    @ATTR(name='axes',     opsets=(1, ), default=()) # v1 add
    @ATTR(name='keepdims', opsets=(1, ), default=1)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReduceSumSquare'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRelu(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Relu'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReshape(AttrBaseOp):

    @ATTR(name='shape',           opsets=(1, 5), default=()) # v1 add, v5 delete
    @ATTR(name='consumed_inputs', opsets=(1, 5), default=()) # v1 add, v5 delete
    def __init__(self, shape, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Reshape'
        check_redundant_attrs(kwargs, self.__class__.__name__)

        # shape from inputs
        self.attributes['shape'] = shape



class AttrResize(AttrBaseOp):

    @ATTR(name='mode',                           opsets=(10, ), default='nearest')            # v10 add
    @ATTR(name='coordinate_transformation_mode', opsets=(11, ), default='half_pixel')         # v10 add
    @ATTR(name='cubic_coeff_a',                  opsets=(11, ), default=-0.75)                # v10 add
    @ATTR(name='exclude_outside',                opsets=(11, ), default=0)                    # v10 add
    @ATTR(name='extrapolation_value',            opsets=(11, ), default=0.0)                  # v10 add
    @ATTR(name='nearest_mode',                   opsets=(11, ), default='round_prefer_floor') # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Resize'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrReverseSequence(AttrBaseOp):

    @ATTR(name='batch_axis', opsets=(10, ), default=1) # v10 add
    @ATTR(name='time_axis',  opsets=(10, ), default=0) # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ReverseSequence'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRoiAlign(AttrBaseOp):

    @ATTR(name='mode',           opsets=(10, ), default='avg') # v10 add
    @ATTR(name='output_height',  opsets=(10, ), default=1)     # v10 add
    @ATTR(name='output_width',   opsets=(10, ), default=1)     # v10 add
    @ATTR(name='sampling_ratio', opsets=(10, ), default=0)     # v10 add
    @ATTR(name='spatial_scale',  opsets=(10, ), default=1.0)   # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'RoiAlign'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrRound(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Round'


class AttrScan(AttrBaseOp):

    @ATTR(name='body',                   opsets=(8, ))              # v8 add, required
    @ATTR(name='directions',             opsets=(8, 9), default=()) # v8 add, v9 delete
    @ATTR(name='num_scan_inputs',        opsets=(8, ))              # v8 add, required
    @ATTR(name='scan_input_axes',        opsets=(9,  ), default=()) # v9 add
    @ATTR(name='scan_input_directions',  opsets=(9,  ), default=()) # v9 add
    @ATTR(name='scan_output_axes',       opsets=(9,  ), default=()) # v9 add
    @ATTR(name='scan_output_directions', opsets=(9,  ), default=()) # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Scan'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrScatter(AttrBaseOp):

    @ATTR(name='axis', opsets=(9, 11), default=0) # v9 add, v11 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Scatter'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrScatterElements(AttrBaseOp):

    @ATTR(name='axis', opsets=(11, ), default=0) # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ScatterElements'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrScatterND(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'ScatterND'


class AttrSelu(AttrBaseOp):

    @ATTR(name='alpha',           opsets=(1,  ), default=1.67326) # v1 add
    @ATTR(name='consumed_inputs', opsets=(1, 6), default=())      # v1 add, v6 delete
    @ATTR(name='gamma',           opsets=(1,  ), default=1.0507)  # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Selu'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSequenceAt(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'SequenceAt'


class AttrSequenceConstruct(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'SequenceConstruct'


class AttrSequenceEmpty(AttrBaseOp):

    @ATTR(name='dtype', opsets=(11,  ), default=-1) # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'SequenceEmpty'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSequenceErase(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'SequenceErase'


class AttrSequenceInsert(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'SequenceInsert'


class AttrSequenceLength(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'SequenceLength'


class AttrShape(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Shape'


class AttrShrink(AttrBaseOp):

    @ATTR(name='bias',  opsets=(9,  ), default=0.0) # v9 add
    @ATTR(name='lambd', opsets=(9,  ), default=0.5) # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Shrink'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSigmoid(AttrBaseOp):

    @ATTR(name='consumed_inputs',    opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Sigmoid'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSign(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Sign'


class AttrSin(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Sin'


class AttrSinh(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Sinh'


class AttrSize(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Size'


class AttrSlice(AttrBaseOp):

    @ATTR(name='axes',      opsets=(1, 10), default=()) # v1 add, v10 delete
    @ATTR(name='ends',      opsets=(1, 10))             # v1 add, required, v10 delete
    @ATTR(name='starts',    opsets=(1, 10))             # v1 add, required, v10 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Slice'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSoftmax(AttrBaseOp):

    @ATTR(name='axis', opsets=( 1, 13), default=1)  # v1 add, v13 delete
    @ATTR(name='axis', opsets=(13,   ), default=-1) # v13 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Softmax'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSoftmaxCrossEntropyLoss(AttrBaseOp):

    @ATTR(name='ignore_index', opsets=(12, ), default=-1)     # v12 add
    @ATTR(name='reduction',    opsets=(12, ), default='mean') # v12 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'SoftmaxCrossEntropyLoss'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSoftplus(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Softplus'


class AttrSoftsign(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Softsign'


class AttrSpaceToDepth(AttrBaseOp):

    @ATTR(name='blocksize', opsets=(1, )) # v1 add, required
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'SpaceToDepth'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSplit(AttrBaseOp):

    @ATTR(name='axis',  opsets=(1,   ), default=0)  # v1 add
    @ATTR(name='split', opsets=(1, 13), default=()) # v1 add, v13 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Split'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSplitToSequence(AttrBaseOp):

    @ATTR(name='axis',     opsets=(11, ), default=0)  # v11 add
    @ATTR(name='keepdims', opsets=(11, ), default=1)  # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'SplitToSequence'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSqrt(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Sqrt'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSqueeze(AttrBaseOp):

    @ATTR(name='axes', opsets=(1, 13), default=()) # v1 add, v13 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Squeeze'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrStringNormalizer(AttrBaseOp):

    @ATTR(name='case_change_action', opsets=(10, ), default='NONE')  # v10 add
    @ATTR(name='is_case_sensitive',  opsets=(10, ), default=0)       # v10 add
    @ATTR(name='locale',             opsets=(10, ), default='en_US') # v10 add
    @ATTR(name='stopwords',          opsets=(10, ), default=())      # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'StringNormalizer'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSub(AttrBaseOp):

    @ATTR(name='axis',            opsets=(1, 7))             # v1 add, v7 delete, required
    @ATTR(name='broadcast',       opsets=(1, 7), default=0)  # v1 add, v7 delete
    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Sub'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrSum(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Sum'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrTan(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Tan'


class AttrTanh(AttrBaseOp):

    @ATTR(name='consumed_inputs', opsets=(1, 6), default=()) # v1 add, v6 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Tanh'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrTfIdfVectorizer(AttrBaseOp):

    @ATTR(name='max_gram_length', opsets=(9, ))             # v9 add, required
    @ATTR(name='max_skip_count',  opsets=(9, ))             # v9 add, required
    @ATTR(name='min_gram_length', opsets=(9, ))             # v9 add, required
    @ATTR(name='mode',            opsets=(9, ))             # v9 add, required
    @ATTR(name='ngram_counts',    opsets=(9, ))             # v9 add, required
    @ATTR(name='ngram_indexes',   opsets=(9, ))             # v9 add, required
    @ATTR(name='pool_int64s',     opsets=(9, ), default=()) # v9 add
    @ATTR(name='pool_strings',    opsets=(9, ), default=()) # v9 add
    @ATTR(name='weights',         opsets=(9, ), default=()) # v9 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'TfIdfVectorizer'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrThresholdedRelu(AttrBaseOp):

    @ATTR(name='alpha', opsets=(10, ), default=1.0) # v10 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'ThresholdedRelu'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrTile(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Tile'


class AttrTopK(AttrBaseOp):

    @ATTR(name='axis',    opsets=( 1,   ), default=-1) # v1 add
    @ATTR(name='k',       opsets=( 1, 10))             # v1 add, required, v10 delete
    @ATTR(name='largest', opsets=(11,   ), default=1)  # v11 add
    @ATTR(name='sorted',  opsets=(11,   ), default=1)  # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'TopK'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrTranspose(AttrBaseOp):

    @ATTR(name='perm', opsets=(1, ), default=()) # v1 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Transpose'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrUnique(AttrBaseOp):

    @ATTR(name='axis',   opsets=(11, ), default=-1) # v11 add
    @ATTR(name='sorted', opsets=(11, ), default=1)  # v11 add
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Unique'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrUnsqueeze(AttrBaseOp):

    @ATTR(name='axes', opsets=(1, 13)) # v11 add, required, v13 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Unsqueeze'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrUpsample(AttrBaseOp):

    @ATTR(name='height_scale', opsets=(1,  7))                    # v1 add, required
    @ATTR(name='width_scale',  opsets=(1,  7))                    # v1 add, required
    @ATTR(name='mode',         opsets=(1, 10), default='nearest') # v1 add, v10 delete
    @ATTR(name='scales',       opsets=(7,  9))                    # v7 add, required, v9 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Upsample'
        check_redundant_attrs(kwargs, self.__class__.__name__)


class AttrWhere(AttrBaseOp):

    def __init__(self, name='', **kwargs):
        self.node_properties['name'] = name
        self.attributes['type'] = 'Where'


class AttrXor(AttrBaseOp):

    @ATTR(name='axis',      opsets=(1, 7), default=-1) # v1 add, v7 delete
    @ATTR(name='broadcast', opsets=(1, 7), default=0)  # v1 add, v7 delete
    def __init__(self, name='', **kwargs):

        self.node_properties['name'] = name
        self.attributes['type'] = 'Xor'
        check_redundant_attrs(kwargs, self.__class__.__name__)