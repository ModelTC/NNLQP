# define the op and its attributes to be encoded
OPS = {
    "Conv": {
        "code": 1,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
            "dilations",
            "group",
            "bias",
        ],
    },
    "Relu": {
        "code": 2,
        "attrs": [],
    },
    "Add": {
        "code": 3,
        "attrs": [],
    },
    "Sigmoid": {
        "code": 4,
        "attrs": [],
    },
    "Reshape": {
        "code": 5,
        "attrs": [],
    },
    "MaxPool": {
        "code": 6,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
            #"dilations",
        ],
    },
    "Split": {
        "code": 7,
        "attrs": [],
    },
    "GlobalAveragePool": {
        "code": 8,
        "attrs": [],
    },
    "Gemm": {
        "code": 9,
        "attrs": [
            "bias",
        ],
    },
    "Transpose": {
        "code": 10,
        "attrs": [
            "perm",
        ]
    },
    "Upsample": {
        "code": 11,
        "attrs": [],
    },
    "BatchNormalization": {
        "code": 12,
        "attrs": [],
    },
    "Mul": {
        "code": 13,
        "attrs": [],
    },
    "Concat": {
        "code": 14,
        "attrs": [],
    },
    "Flatten": {
        "code": 15,
        "attrs": [],
    },
    "AveragePool": {
        "code": 16,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
        ],
    },
    "Cast": {
        "code": 17,
        "attrs": [],
    },
    "Matmul": {
        "code": 18,
        "attrs": [],
    },
    "ReduceMean": {
        "code": 19,
        "attrs": [],
    },
    "Pow": {
        "code": 20,
        "attrs": [],
    },
    "Slice": {
        "code": 21,
        "attrs": [],
    },
    "Div": {
        "code": 22,
        "attrs": [],
    },
    "Sub": {
        "code": 23,
        "attrs": [],
    },
    "Sqrt": {
        "code": 24,
        "attrs": [],
    },
    "Clip": {
        "code": 25,
        "attrs": [],
    },
    "Softmax": {
        "code": 26,
        "attrs": [],
    },
    "Tanh": {
        "code": 27,
        "attrs": [],
    },
    "ConvTranspose": {
        "code": 28,
        "attrs": [
            "kernel_shape",
            "strides",
            "pads",
            "dilations",
            "group",
            "bias",
            "output_padding",
        ],
    },
}

# define the value type of attr value, and its feature length
ATTRS = {
    "kernel_shape"  : ("tuple", 1,  2.53,   56),
    "strides"       : ("tuple", 1,  1.16,   56),
    "pads"          : ("tuple", 1,  0.75,    6),
    "dilations"     : ("tuple", 1,  1.00,    1),
    "group"         : ("int"  ,    15.08, 6144),
    "bias"          : ("bool" ,     0.77,    1),
    "perm"          : ("tuple", 8,  1.78,    4),
    "output_padding": ("tuple", 1,  0.00, 1e-5),
}

# define the fixed length of feature op_code, attrs, output shape
FEATURE_LENGTH = {
    "op_code": 32,
    "attrs": 8,
    "output_shape": 4,
}
