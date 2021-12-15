import os
import onnx
import time
import torch
import random
import numpy as np
from torch_geometric.data import Data, Dataset
from feature.graph_feature import extract_graph_feature


def get_torch_data(onnx_file, batch_size, cost_time):
    adjacent, node_features, static_features = extract_graph_feature(onnx_file, batch_size)
    edge_index = torch.from_numpy(np.array(np.where(adjacent > 0))).type(torch.long)
    node_features = np.array(node_features, dtype=np.float32)
    x = torch.from_numpy(node_features).type(torch.float)
    sf = torch.from_numpy(static_features).type(torch.float)
    y = torch.FloatTensor([cost_time])
    data = Data(
        x = x,
        edge_index = edge_index,
        y = y,
    )
    return data, sf


class GraphLatencyDataset(Dataset):
    # specific a platform
    def __init__(self, root, onnx_dir, latency_file, override_data=False, transform=None, pre_transform=None,
                model_types=None, train_test_stage=None, platforms=None, sample_num=-1):
        super(GraphLatencyDataset, self).__init__(root, transform, pre_transform)
        self.onnx_dir = onnx_dir
        self.latency_file = latency_file
        self.latency_ids = []
        self.override_data = override_data
        self.model_types = model_types
        self.train_test_stage = train_test_stage
        self.platforms = platforms

        # Load the gnn model for block
        self.device = None
        print("Extract input data from onnx...")
        self.custom_process()
        print("Done.")

        if sample_num > 0:
            random.seed(1234)
            random.shuffle(self.latency_ids)
            self.latency_ids = self.latency_ids[:sample_num]
        random.seed(1234)
        random.shuffle(self.latency_ids)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def custom_process(self):
        with open(self.latency_file) as f:
            for line in f.readlines():

                line = line.rstrip()
                items = line.split(" ")
                speed_id = str(items[0])
                graph_id = str(items[1])
                batch_size = int(items[2])
                cost_time = float(items[3])
                plt_id = int(items[5])

                if self.model_types and items[4] not in self.model_types:
                    continue

                if self.platforms and plt_id not in self.platforms:
                    continue

                if self.train_test_stage and items[6] != self.train_test_stage:
                    continue

                onnx_file = os.path.join(self.onnx_dir, graph_id)
                if os.path.exists(onnx_file):
                    data_file = os.path.join(self.processed_dir, '{}_{}_data.pt'.format(speed_id, plt_id))
                    sf_file = os.path.join(self.processed_dir, '{}_{}_sf.pt'.format(speed_id, plt_id))
                    graph_name = "{}_{}_{}".format(graph_id, batch_size, plt_id)
                    self.latency_ids.append((data_file, sf_file, graph_name, plt_id))

                    if (not self.override_data) and os.path.exists(data_file) and os.path.exists(sf_file):
                        continue

                    if len(self.latency_ids) % 1000 == 0:
                        print(len(self.latency_ids))

                    try:
                        GG = onnx.load(onnx_file)
                        data, sf = get_torch_data(GG, batch_size, cost_time)

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        torch.save(data, data_file)
                        torch.save(sf, sf_file)
                    except Exception as e:
                        self.latency_ids.pop()
                        print("Error", e)

    def len(self):
        return len(self.latency_ids)

    def get(self, idx):
        data_file, sf_file, graph_name, plt_id = self.latency_ids[idx]
        data = torch.load(data_file)
        sf = torch.load(sf_file)
        return data, sf, graph_name, plt_id