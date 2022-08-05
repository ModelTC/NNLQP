from genericpath import exists
import sys
import os
import time
import torch
import logging
import argparse
import numpy as np

from torch_geometric.data import Data, Batch
from predictor.feature.graph_feature import extract_graph_feature


def download_file(url, file_name):
    if not os.path.exists(file_name):
        import urllib.request as urllib2
        try:
            print("download from {} to {}...".format(url, file_name))
            if sys.version_info >= (3,):
                urllib2.urlretrieve(url, file_name)
            else:
                f = urllib2.urlopen(url)
                data = f.read()
                with open(file_name, "wb") as code:
                    code.write(data)
        except Exception as err:
            if os.path.exists(file_name):
                    os.remove(file_name)
            raise Exception("download {} failed due to {}!".format(file_name, repr(err)))


class Demo(object):

    def __init__(self):
        self.args = self.init_args()
        print("Loading args: \n{}".format(self.args))
        self.model = self.init_model()
        # print("Loading model: \n{}".format(self.model))

    def init_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cpu', action='store_true')
        parser.add_argument('--onnx_file', required=True, type=str)
        args = parser.parse_args()

        args.cpu = True if args.cpu else False
        args.batch_size = 8
        args.multi_plt = "2,9,10,12,13,14,16,18,23"
        args.multi_names = [
            'cpu-openppl-fp32', 'hi3559A-nnie11-int8', 'gpu-T4-trt7.1-fp32',
            'gpu-T4-trt7.1-int8', 'gpu-P4-trt7.1-fp32', 'gpu-P4-trt7.1-int8',
            'hi3519A-nnie12-int8', 'atlas300-acl-fp16', 'mul270-neuware-int8',
        ]

        args.multi_plt = {int(x): k for k, x in enumerate(args.multi_plt.split(','))} if args.multi_plt else {}
        ckpt_dir = os.path.join("experiments", "multi_platform")
        args.resume = os.path.join(ckpt_dir, "checkpoints", "plt_2_9_10_12_13_14_16_18_23", "ckpt_best.pth")

        # download ckpt from the Internet
        if not os.path.exists(args.resume):
            fname = os.path.join(ckpt_dir, "multi_platform_ckpt.tar.gz")
            download_file("https://github.com/ModelTC/NNLQP/releases/download/v1.0-data/multi_platform_ckpt.tar.gz", fname)
            os.system("tar -xzvf {} -C ./experiments/multi_platform/".format(fname))
        return args

    def init_model(self):
        from predictor.model import Net
        model = Net(multi_plt=self.args.multi_plt)
        if self.args.resume:
            print("Loading checkpoint: {}".format(self.args.resume))
            ckpt = torch.load(self.args.resume)
            start_epoch, best_acc = ckpt['epoch'], ckpt['best_acc']
            model.load_state_dict(ckpt['state_dict'], strict = True)
            print("loaded checkpoint: {} (epoch {} best {:.2f})". \
                             format(self.args.resume, start_epoch, best_acc))
        return model

    def inference_once(self, onnx_file, batch_size, device):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model = self.model.to(device)
        self.model.eval()
        t0 = time.time()

        # extract feature, to torch data
        A, F, S = extract_graph_feature(onnx_file, batch_size)

        F = np.array(F, dtype=np.float32)
        x = torch.from_numpy(F).type(torch.float)
        E = torch.from_numpy(np.array(np.where(A > 0))).type(torch.long)
        S = torch.from_numpy(S).type(torch.float)

        batch = Batch()
        batch = batch.from_data_list([Data(x = x, edge_index=E)])
        batch = batch.to(device)
        static_feature = S.to(device).view(1, -1)

        preds = self.model(batch, static_feature)
        preds = preds.view(-1).data.cpu().numpy()
        t1 = time.time()
        print("Model inference cost: {} ms".format((t1 - t0) * 1000))

        if self.args.multi_plt:
            for k, v in self.args.multi_plt.items():
                print("Latency prediction for platform {} : {} ms".format(self.args.multi_names[v], preds[v]))
        else:
            print("Latency prediction: {} ms".format(preds[0]))

    def run(self):
        device = torch.device('cuda' if not self.args.cpu and torch.cuda.is_available() else 'cpu')
        self.inference_once(self.args.onnx_file, self.args.batch_size, device)


if __name__ == "__main__":
    x = Demo()
    x.run()