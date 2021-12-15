import numpy as np
from .feature_utils import OPS, ATTRS, FEATURE_LENGTH


# int -> one hot embedding
def embed_op_code(op_type):
    length = FEATURE_LENGTH["op_code"]
    if op_type not in OPS:
        return np.zeros(length, dtype="float32")
    op_code = OPS[op_type]["code"] - 1
    if op_code >= length:
        raise Exception("op code of {}: {} greater than one-hot length {}!".format(
            op_type, op_code, length))
    return np.eye(length, dtype="float32")[op_code]


class EmbedValue:
    # int value embedding
    @staticmethod
    def embed_int(x, center=0, scale=1):
        x = np.array([int(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    # float value embedding
    @staticmethod
    def embed_float(x, center=0, scale=1):
        x = np.array([float(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    # bool value embedding
    @staticmethod
    def embed_bool(x, center=0, scale=1):
        x = np.array([int(bool(x))], dtype="float32")
        return (x - center) / np.abs(scale)

    # tuple value embedding
    @staticmethod
    def embed_tuple(x, length, center=0, scale=1):
        x = np.array(x, dtype="float32").reshape(-1)
        if x.size > length:
            x = x[:length]
        if x.size < length:
            x = np.concatenate([x, np.zeros(length - x.size, dtype="float32")])
        if not isinstance(center, list):
            center = [center] * x.size
        if not isinstance(scale, list):
            scale = [scale] * x.size
        center = np.array(center, dtype="float32")
        scale = np.array(scale, dtype="float32")
        return (x - center) / np.abs(scale)


# attrs embedding
def embed_attrs(op_type, attrs):
    length = FEATURE_LENGTH["attrs"]
    if op_type not in OPS:
        return np.zeros(length, dtype="float32")

    feats = []
    for name in OPS[op_type]["attrs"]:
        assert name in attrs, "attr {} for {} need to be encoded but not included!".format(name, op_type)
        assert name in ATTRS, "attr {} for {} does not defined in ATTRS!".format(name, op_type)

        attr_value = attrs[name]
        attr_def = ATTRS[name]
        feat = getattr(EmbedValue, "embed_" + attr_def[0])(attr_value, *attr_def[1:])
        feats.append(feat)

    # concat attr features
    feats = np.concatenate(feats) if len(feats) > 0 else np.zeros(length, dtype="float32")
    feat_len = feats.size
    if feat_len > length:
        raise Exception("tuple length {} is grater than the embed length {}".format(
            feat_len, length))
    if feat_len < length:
        feats = np.concatenate([feats, np.zeros(length - feat_len, dtype="float32")])
    return feats


# networkx_G -> op_code_embeddings & attrs_embeddings
# output_shapes -> output_shape_embeddings
def extract_node_features(networkx_G, output_shapes, batch_size):
    embeddings = {}

    for node in networkx_G.nodes.data():
        attrs = node[1]["attr"].attributes
        node_name = node[0]
        op_type = attrs["type"]

        # one hot op_code embedding
        op_code_embedding = embed_op_code(op_type)

        # fixed length embedding for attrs, need normalize?
        attrs_embedding = embed_attrs(op_type, attrs)

        # fixed length embedding for output shape, need normalize?
        assert node_name in output_shapes, "could not find output shape for node {}".format(node_name)
        output_shape_embedding = EmbedValue.embed_tuple(
            output_shapes[node_name],
            FEATURE_LENGTH["output_shape"],
            [12.25, 286.96, 32.62, 31.80],
            [1024, 2496, 256, 256],
            #[11.02, 228.78, 48.94, 63.10],
            #[28.38, 642.21, 78.54, 93.87],
        )

        # concat to the final node feature
        embeddings[node_name] = np.concatenate([
            op_code_embedding,
            attrs_embedding,
            output_shape_embedding,
        ])
        # print(op_type, len(embeddings[node_name]), embeddings[node_name])

    return embeddings
