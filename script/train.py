from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from utils import CellImageLoad, CellImageLoadTest, CellImageLoadPseudo, set_seed, VisdomClass
from networks import UNet
import numpy as np
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
from networks import MaskMSELoss
import argparse

seed = 1


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id):
    random.seed(worker_id + seed)
    np.random.seed(worker_id + seed)


def gather_path(train_paths, mode, extension):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob(extension)))
    return ori_paths


class TrainNet(VisdomClass):
    def __init__(self, args):
        if args.pseudo:
            data_loader = CellImageLoadPseudo(
                args.imgs, args.gts, args.bg_masks, args.dataset, args.crop_size, args.crop)
        else:
            data_loader = CellImageLoad(args.imgs, args.gt, args.dataset, args.crop_size, args.crop)

        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)

        self.number_of_traindata = data_loader.__len__()

        self.save_weight_path = args.weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_weight_path.parent.joinpath("epoch_weight").mkdir(parents=True, exist_ok=True)
        print("Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
            args.epochs, args.batch_size, args.learning_rate, args.gpu))

        self.net = args.net

        self.multi_gpu = args.multi_gpu
        self.iter_max = args.iter_max

        self.vis = args.vis
        self.env = args.dataset + str(args.base_frame)
        self.dataset = args.dataset
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.device = args.device
        if args.pseudo:
            self.criterion = MaskMSELoss()
        else:
            self.criterion = nn.MSELoss()
        self.losses = []
        self.epoch_loss = 0
        self.pseudo = args.pseudo

    def show_graph(self):
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        plt.savefig(str(self.save_weight_path.parent.joinpath("loss_curce.png")))
        plt.close()

    def main(self):
        if self.vis:
            self.vis_init()
        iteration = 0
        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))
            self.net.train()
            pbar = tqdm(total=self.number_of_traindata)
            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks = data["gt"]
                if self.pseudo:
                    bg_mask = data["bg"]
                if self.gpu:
                    imgs = imgs.to(self.device)
                    true_masks = true_masks.to(self.device)
                    if self.pseudo:
                        bg_mask = bg_mask.to(self.device)

                mask_preds = self.net(imgs)

                if self.pseudo:
                    loss = self.criterion(mask_preds, true_masks, bg_mask)
                else:
                    loss = self.criterion(mask_preds, true_masks)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iteration += 1

                if self.vis:
                    if not self.pseudo:
                        bg_mask = None
                    self.vis_show_result(iteration, loss, mask_preds, imgs, true_masks, bg_mask)

                if iteration % 3000 == 0:
                    if self.multi_gpu:
                        torch.save(self.net.module.state_dict(),
                                   str(self.save_weight_path.parent.joinpath(f"epoch_weight/{epoch:05d}.pth")))
                    else:
                        torch.save(self.net.state_dict(),
                                   str(self.save_weight_path.parent.joinpath(f"epoch_weight/{epoch:05d}.pth")))
                if iteration > self.iter_max:
                    break
                pbar.update(self.batch_size)

            pbar.close()
            loss = self.epoch_loss / (self.number_of_traindata + 1)
            print("Epoch finished ! Loss: {}".format(loss))
            self.losses.append(loss)
            self.epoch_loss = 0

            if iteration > self.iter_max:
                print("stop running")
                break
        if self.multi_gpu:
            torch.save(self.net.module.state_dict(), str(self.save_weight_path))
        else:
            torch.save(self.net.state_dict(), str(self.save_weight_path))

        self.show_graph()


CTC_DATASET = ["DIC-C2DH-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC"]


def call(args):
    args.net = UNet(n_channels=1, n_classes=1, norm=args.norm)

    if args.multi_gpu:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            args.net = nn.DataParallel(args.net)
    if args.gpu:
        args.net.to(args.device)

    train = TrainNet(args)
    train.main()


def pseudo_train(args, pseudo_path, weight_path_base):
    args.norm = "instance"
    bg_label = ""
    if args.dataset in CTC_DATASET:
        seq_list = ["01/", "02/"]
    else:
        seq_list = [""]
    for args.seq in seq_list:
        args.bg_masks = sorted(pseudo_path.joinpath(f"{args.seq}bg_pseudo{bg_label}").glob("*.png"))
        args.imgs = sorted(pseudo_path.joinpath(f"{args.seq}img").glob("*.png"))
        args.gts = sorted(pseudo_path.joinpath(f"{args.seq}fg_pseudo").glob("*.png"))

        if (len(args.imgs) == 0) or (len(args.imgs) != len(args.gts)) or (len(args.imgs) != len(args.bg_masks)):
            print("No additional pseudo labels")
            continue

        args.vis = False
        args.val = False
        args.pseudo = True
        args.weight_path = weight_path_base.joinpath(f"{args.seq}final.pth")
        call(args)


def supervised_one(args, weight_path_base):
    args.norm = "instance"
    seq_list = [""]
    if args.dataset in CTC_DATASET:
        seq_list = ["01/", "02/"]

    for args.seq in seq_list:
        base_path = Path(f"image/{args.dataset}")
        if args.dataset in CTC_DATASET:
            args.imgs = sorted(base_path.joinpath(f"{args.seq}img").glob("*.png"))
            args.gt = sorted(base_path.joinpath(f"{args.seq}gt").glob("*.png"))
            args.gt = [args.gt[args.base_frame]]
        else:
            args.imgs = sorted(base_path.joinpath(f"train/img").glob("*.*"))
            args.gt = sorted(base_path.joinpath(f"train/gt").glob("*.*"))

        if args.dataset == "PhC-C2DL-PSC":
            args.imgs = args.imgs[150:250]
        args.imgs = [args.imgs[args.base_frame]]

        args.pseudo = False
        args.weight_path = weight_path_base.joinpath(f"{args.seq}final.pth")
        call(args)
