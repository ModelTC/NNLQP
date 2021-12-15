
import os
import time
import copy
import random
import torch
import logging
import argparse
import numpy as np

import torch.nn.functional as F
from torch_geometric.data import DataLoader
from dataset import GraphLatencyDataset


class Metric(object):
    def __init__(self):
        self.all = self.init_pack()
        self.plts = {}

    def init_pack(self):
        return {
            'cnt': 0,
            'apes': [],                                # absolute percentage error
            'errbnd_cnt': np.array([0.0, 0.0, 0.0]),   # error bound count
            'errbnd_val': np.array([0.1, 0.05, 0.01]), # error bound value: 0.1, 0.05, 0.01
        }

    def update_pack(self, ps, gs, pack):
        for i in range(len(ps)):
            ape = np.abs(ps[i] - gs[i]) / gs[i]
            pack['errbnd_cnt'][ape <= pack['errbnd_val']] += 1
            pack['apes'].append(ape)
        pack['cnt'] += len(ps)

    def measure_pack(self, pack):
        acc = np.mean(pack['apes'])
        err = (pack['errbnd_cnt'] / pack['cnt'])[0]
        return acc, err, pack['cnt']

    def update(self, ps, gs, plts=None):
        self.update_pack(ps, gs, self.all)
        if plts:
            for idx, plt in enumerate(plts):
                if plt not in self.plts:
                    self.plts[plt] = self.init_pack()
                self.update_pack([ps[idx]], [gs[idx]], self.plts[plt])

    def get(self, plt=None):
        if plt is None:
            return self.measure_pack(self.all)
        else:
            return self.measure_pack(self.plts[plt])


class Trainer(object):

    def __init__(self):
        self.args = self.init_args()
        self.logger = self.init_logger()        
        self.logger.info("Loading args: \n{}".format(self.args))

        if self.args.gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.args.gpu)
            if not torch.cuda.is_available():
                self.logger.error("No GPU={} found!".format(self.args.gpu))

        self.logger.info("Loading dataset:")
        if self.args.train_test_stage:
            self.train_loader, self.test_loader = self.init_train_test_dataset()
        else:
            self.train_loader, self.test_loader = self.init_unseen_structure_dataset()

        self.logger.info("Loading model:")
        self.model, self.start_epoch, self.best_acc = self.init_model()
        self.best_err = 0
        # print("Model:", self.model)

        self.device = torch.device('cuda' if self.args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)

    def init_args(self):        
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--gpu', type=int, default=0, help="gpu id, < 0 means no gpu")
        parser.add_argument('--steps', type=str, default="", help='learning decay')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)

        parser.add_argument('--gnn_layer', type=str)
        parser.add_argument('--data_root', type=str)
        parser.add_argument('--all_latency_file', type=str)
        parser.add_argument('--test_model_type', type=str)
        parser.add_argument('--train_model_types', type=str)
        parser.add_argument('--train_test_stage', action='store_true')
        parser.add_argument('--norm_sf', action='store_true')
        parser.add_argument('--train_num', type=int, default=-1)

        parser.add_argument('--onnx_dir', type=str)
        parser.add_argument('--override_data', action='store_true')

        parser.add_argument('--log', type=str)
        parser.add_argument('--pretrain', type=str)
        parser.add_argument('--resume', type=str)
        parser.add_argument('--model_dir', type=str)

        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--ckpt_save_freq', type=int, default=20)
        parser.add_argument('--test_freq', type=int, default=10)
        parser.add_argument('--only_test', action='store_true')
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--num_node_features', type=int, default=44)
        parser.add_argument('--multi_plt', type=str, default="")

        args = parser.parse_args()
        args.steps = [int(x) for x in args.steps.split(',')] if args.steps else []
        args.norm_sf = True if args.norm_sf else False
        # {plt_id: order_id}
        args.multi_plt = {int(x): k for k, x in enumerate(args.multi_plt.split(','))} if args.multi_plt else {}
        return args


    def init_logger(self):
        if not os.path.exists("log"):
            os.makedirs("log")

        logger = logging.getLogger("FEID")
        logger.setLevel(level = logging.INFO)
        formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d" \
                                      "-%(levelname)s-%(message)s")

        # log file stream
        handler = logging.FileHandler(self.args.log)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)

        # log console stream
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
    
        logger.addHandler(handler)
        logger.addHandler(console)
    
        return logger


    def init_unseen_structure_dataset(self):
        model_types = set()
        for line in open(self.args.all_latency_file).readlines():
            model_types.add(line.split()[4])
        assert self.args.test_model_type in model_types
        test_model_types = set([self.args.test_model_type])
        if self.args.train_model_types:
            train_model_types = set(self.args.train_model_types.split(','))
            train_model_types = train_model_types & model_types
        else:
            train_model_types = model_types - test_model_types
        assert len(train_model_types) > 0
        allow_platforms=self.args.multi_plt.keys() if self.args.multi_plt else None

        self.logger.info("Train model types: {}".format(train_model_types))
        self.logger.info("Test model types: {}".format(test_model_types))
        self.logger.info("Platforms: {}".format(allow_platforms))

        train_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            override_data=self.args.override_data,
            model_types=train_model_types,
            platforms=allow_platforms,
        )
        test_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            override_data=self.args.override_data,
            model_types=test_model_types,
            platforms=allow_platforms,
        )

        self.logger.info("Train data = {}, Test data = {}".format(len(train_set), len(test_set)))
        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)
        return train_loader, test_loader


    def init_train_test_dataset(self):
        allow_platforms=self.args.multi_plt.keys()
        train_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            override_data=self.args.override_data,
            train_test_stage="train",
            platforms=allow_platforms,
            sample_num=self.args.train_num,
        )
        test_set = GraphLatencyDataset(
            self.args.data_root,
            self.args.onnx_dir,
            self.args.all_latency_file,
            override_data=self.args.override_data,
            train_test_stage="test",
            platforms=allow_platforms,
        )

        self.logger.info("Train data = {}, Test data = {} for platforms {}".format(
            len(train_set), len(test_set), allow_platforms
        ))
        self.logger.info("Platforms: {}".format(allow_platforms))
        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)
        return train_loader, test_loader


    def init_model(self):
        best_acc = 1e9
        start_epoch = 1

        from model import Net
        model = Net(
            num_node_features=self.args.num_node_features,
            gnn_layer=self.args.gnn_layer,
            gnn_hidden=self.args.hidden_size,
            fc_hidden=self.args.hidden_size,
            multi_plt=self.args.multi_plt,
            norm_sf=self.args.norm_sf,
        )

        if self.args.pretrain:
            self.logger.info("Loading pretrain: {}".format(self.args.pretrain))
            ckpt = torch.load(self.args.pretrain)
            model.load_state_dict(ckpt['state_dict'], strict = False)
            self.logger.info("Loaded pretrain: {}".format(self.args.pretrain))

        if self.args.resume:
            self.logger.info("Loading checkpoint: {}".format(self.args.resume))
            ckpt = torch.load(self.args.resume)
            start_epoch, best_acc = ckpt['epoch'], ckpt['best_acc']
            model.load_state_dict(ckpt['state_dict'], strict = True)
            self.logger.info("loaded checkpoint: {} (epoch {} best {:.2f})". \
                             format(self.args.resume, start_epoch, best_acc))

        return model, start_epoch, best_acc


    def adjust_learning_rate(self, epoch):
        ind = len(list(filter(lambda x: x <= epoch, self.args.steps)))
        lr = self.args.lr * (0.1 ** ind)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


    def format_second(self, secs):
        return "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format( \
               int(secs / 3600), int((secs % 3600) / 60), int(secs % 60))
    

    def save_checkpoint(self, epoch, best = False):
        epoch_str = "best" if best else "e{}".format(epoch)
        model_path = "{}/ckpt_{}.pth".format(self.args.model_dir, epoch_str)

        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
        }, model_path)
        self.logger.info("Checkpoint saved to {}".format(model_path))
        return


    def train_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        metric = Metric()

        lr = self.adjust_learning_rate(epoch)
        num_iter = len(self.train_loader)

        for iteration, batch in enumerate(self.train_loader):
            torch.cuda.empty_cache()

            data, static_feature, _, plt_id = batch
            data.y = data.y.view(-1, 1)
            data = data.to(self.device)
            static_feature = static_feature.to(self.device)

            self.optimizer.zero_grad()
            pred_cost = self.model(data, static_feature)

            # multi plt training, gather the specific out channel
            if len(self.args.multi_plt) > 1:
                gather_ids = torch.LongTensor([self.args.multi_plt[int(x)] for x in plt_id]).view(-1, 1)
                gather_ids = gather_ids.to(self.device)
                pred_cost = torch.gather(pred_cost, 1, gather_ids)

            loss = F.mse_loss(pred_cost / data.y, data.y / data.y)
            loss.backward()
            self.optimizer.step()

            ps = pred_cost.data.cpu().numpy()[:, 0].tolist()
            gs = data.y.data.cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)
            acc, err, cnt = metric.get()

            if iteration % self.args.print_freq ==  0:
                t1 = time.time()
                speed = (t1 - t0) / (iteration + 1)
                exp_time = self.format_second(speed * (num_iter * (self.args.epochs - epoch + 1) - iteration))

                self.logger.info("Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} MAPE:{:.5f} " \
                                "ErrBnd(0.1):{:.5f} Speed:{:.2f} ms/iter {}" .format( \
                                epoch, self.args.epochs, iteration, num_iter, lr, loss.data, acc, \
                                err, speed * 1000, exp_time))
        return acc


    def test(self):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model.eval()
        t0 = time.time()
        num_iter = len(self.test_loader)
        if num_iter <= 0:
            return 0, 0
        metric = Metric()

        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                torch.cuda.empty_cache()

                data, static_feature, graph_name, plt_id = batch
                data.y = data.y.view(-1, 1)
                data = data.to(self.device)
                static_feature = static_feature.to(self.device)
                pred_cost = self.model(data, static_feature)

                # multi plt training, gather the specific out channel
                if len(self.args.multi_plt) > 1:
                    gather_ids = torch.LongTensor([self.args.multi_plt[int(x)] for x in plt_id]).view(-1, 1)
                    gather_ids = gather_ids.to(self.device)
                    pred_cost = torch.gather(pred_cost, 1, gather_ids)

                #pred_cost = torch.exp(pred_cost)
                ps = pred_cost.data.cpu().numpy()[:, 0].tolist()
                gs = data.y.data.cpu().numpy()[:, 0].tolist()
                plts = plt_id.data.cpu().numpy().tolist()
                metric.update(ps, gs, plts)
                acc, err, cnt = metric.get()

                if iteration > 0 and iteration % 50 == 0:
                    self.logger.info("[{}/{}] MAPE: {:.5f} ErrBnd(0.1): {:.5f}".format(
                        iteration, num_iter, acc, err))

            t1 = time.time()
            speed = (t1 - t0) / num_iter * 1000
            acc, err, cnt = metric.get()

            self.logger.info(" ------------------------------------------------------------------")
            self.logger.info(" * Speed: {:.5f} ms/iter".format(speed))
            self.logger.info(" * MAPE: {:.5f}".format(acc))
            self.logger.info(" * ErrorBound (0.1): {}".format(err))
            self.logger.info(" ------------------------------------------------------------------")

            if self.args.multi_plt:
                accs, errs = [], []
                for plt in self.args.multi_plt.keys():
                    acc, err, cnt = metric.get(plt=plt)
                    accs.append(acc)
                    errs.append(err)
                    self.logger.info("Platform {}, MAPE={:.5f}, ErrorBound(0.1)={:.5f}".format(plt, acc, err))
                self.logger.info(" ------------------------------------------------------------------")
                self.logger.info(" * Platform Average")
                self.logger.info(" * MAPE: {:.5f}".format(np.mean(accs)))
                self.logger.info(" * ErrorBound (0.1): {}".format(np.mean(errs)))
                self.logger.info(" ------------------------------------------------------------------")

        torch.manual_seed(time.time())
        torch.cuda.manual_seed_all(time.time())
        return acc, err


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.args.only_test = False
            self.train_epoch(epoch)

            if epoch > 0 and epoch % self.args.ckpt_save_freq == 0:
                self.save_checkpoint(epoch)

            if epoch > 0 and epoch % self.args.test_freq == 0:
                self.args.only_test = True
                acc, err = self.test()

                if acc < self.best_acc:
                    self.best_acc = acc
                    self.best_err = err
                    self.save_checkpoint(epoch, best = True)
        self.logger.info("Train over, best acc = {:.5f}, err = {}".format(
            self.best_acc, self.best_err
        ))
        return


    def run(self):
        if self.args.only_test:
            self.test()
        else:
            self.train()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
