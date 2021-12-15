import math
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter


def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return
    if init_type =='thomas':
        size = tensor.size(-1)
        stdv = 1. / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == 'kaiming_normal_in':
        nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_normal_out':
        nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_in':
        nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
    elif init_type == 'kaiming_uniform_out':
        nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
    elif init_type == 'orthogonal':
        nn.init.orthogonal_(tensor, gain=nn.init.calculate_gain(nonlinearity))
    else:
        raise ValueError(f'Unknown initialization type: {init_type}')


class PredictFC(torch.nn.Module):
    def __init__(self,
                 input_feature,
                 fc_hidden):
        super(PredictFC, self).__init__()
        self.fc_1 = nn.Linear(input_feature,  fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_relu_1 = nn.ReLU()
        self.fc_relu_2 = nn.ReLU()
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.predictor = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_relu_1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu_2(x)
        x = self.fc_drop_2(x)
        x = self.predictor(x)
        return x


# reduce_func: "sum", "mul", "mean", "min", "max"
class Net(torch.nn.Module):
    def __init__(
        self,
        num_node_features=44,
        gnn_layer="SAGEConv",
        gnn_hidden=512,
        fc_hidden=512,
        reduce_func="sum",
        multi_plt={},
        norm_sf=False,
    ):
        super(Net, self).__init__()

        self.reduce_func = reduce_func
        self.num_node_features = num_node_features
        self.multi_plt = multi_plt
        self.norm_sf = norm_sf
        self.gnn_layer_func = getattr(tgnn, gnn_layer)

        self.graph_conv_1 = self.gnn_layer_func(num_node_features, gnn_hidden, normalize=True)
        self.graph_conv_2 = self.gnn_layer_func(gnn_hidden, gnn_hidden, normalize=True)
        self.gnn_drop_1 = nn.Dropout(p=0.05)
        self.gnn_drop_2 = nn.Dropout(p=0.05)
        self.gnn_relu1 = nn.ReLU()
        self.gnn_relu2 = nn.ReLU()

        out_dim = 1 if len(multi_plt) <= 1 else len(multi_plt)
        if out_dim > 1:
            for i in range(out_dim):
                self.add_module(f'head_{i}', PredictFC(gnn_hidden + 4, fc_hidden))
        else:
            if self.norm_sf:
                self.norm_sf_linear = nn.Linear(4, gnn_hidden)
                self.norm_sf_drop = nn.Dropout(p=0.05)
                self.norm_sf_relu = nn.ReLU()
                sf_hidden = gnn_hidden
            else:
                sf_hidden = 4
            self.fc_1 = nn.Linear(gnn_hidden + sf_hidden, fc_hidden)
            self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
            self.fc_drop_1 = nn.Dropout(p=0.05)
            self.fc_drop_2 = nn.Dropout(p=0.05)
            self.fc_relu1 = nn.ReLU()
            self.fc_relu2 = nn.ReLU()
            self.predictor = nn.Linear(fc_hidden, out_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas", "relu")
                init_tensor(m.bias, "thomas", "relu")
            elif isinstance(m, self.gnn_layer_func):
                pass

    def forward(self, data, static_feature):
        x, A = data.x, data.edge_index
        x = self.graph_conv_1(x, A)
        x = self.gnn_relu1(x)
        x = self.gnn_drop_1(x)

        x = self.graph_conv_2(x, A)
        x = self.gnn_relu2(x)
        x = self.gnn_drop_2(x)

        gnn_feat = scatter(x, data.batch, dim=0, reduce=self.reduce_func)
        if self.norm_sf:
            static_feature = self.norm_sf_linear(static_feature)
            static_feature = self.norm_sf_drop(static_feature)
            static_feature = self.norm_sf_relu(static_feature)
        x = torch.cat([gnn_feat, static_feature], dim=1)

        if len(self.multi_plt) > 1:
            xs = []
            for i in range(len(self.multi_plt)):
                predictor = getattr(self, f'head_{i}')
                z = predictor(x)
                xs.append(z)
            x = torch.cat(xs, dim=1)
        else:
            x = self.fc_1(x)
            x = self.fc_relu1(x)
            x = self.fc_drop_1(x)
            x = self.fc_2(x)
            x = self.fc_relu2(x)
            feat = self.fc_drop_2(x)
            x = self.predictor(feat)

        pred = -F.logsigmoid(x) # (0, +inf)
        return pred