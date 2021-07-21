from pathlib import Path
import datetime
import torch
import numpy as np
import random
import os
import argparse

from script.ctc_preprocessing import preprocessing_img
from script.train import pseudo_train, supervised_one
from script.predict import pred, pred_test
from script.generate_pseudo_label import pseudo_label_gen, pseudo_label_gen_step2

from utils import set_seed


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("-d", "--data_path", dest="data_path", help="dataset", default="./image/CTC", type=str)
    parser.add_argument("--multi", dest="multi_gpu", help="muti gpu", default=False, type=bool)
    parser.add_argument("-g", "--gpu", dest="gpu", help="whether use CUDA", type=bool, default=True)
    parser.add_argument("-dn", "--device", dest="device", help="select gpu device", default=0, type=int)
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="batch_size", default=8, type=int)
    parser.add_argument("-e", "--epochs", dest="epochs", help="num of epochs", default=10000, type=int)
    parser.add_argument("--iter_max", dest="iter_max", help="maximu number of iteration", default=10000, type=int)
    parser.add_argument("-l", "--learning_rate", dest="learning_rate", help="learning late", default=1e-3, type=float)
    parser.add_argument("-cr", "--crop", dest="crop", help="crop mode", default=False, type=bool)
    parser.add_argument("--crop_size", dest="crop_size", help="crop size", default=(256, 256), type=tuple)
    parser.add_argument("--norm", dest="norm", help="normalization", default="instance", type=str)
    parser.add_argument("--visdom", dest="vis", help="visdom show", default=False, type=bool)
    parser.add_argument("--pseudo", dest="pseudo", help="pseudo flag", default=False, type=bool)
    parser.add_argument("-lf", "--base_frame", dest="labeled frame", default=10, type=int)
    parser.add_argument("-dth", "--dist_th", dest="dist_th", help="distance thresh for detection", default=15, type=int)
    parser.add_argument("--f_rate", dest="rate_of_failure", help="", default=0.8, type=float)

    args = parser.parse_args()
    return args


DATASETS = {
    1: "DIC-C2DH-HeLa",
    2: "PhC-C2DH-U373",
    # 3: "PhC-C2DL-PSC",
    # 4: "Control",
    # 5: "FGF2",
    # 6: "BMP2",
    # 7: "BMP2+FGF2",
}

if __name__ == "__main__":
    params = parse_args()

    # preprocessing_img(DATASETS, params.data_path)
    for params.dataset in DATASETS.values():
        set_seed(1)

        if params.dataset in ["BMP2", "Control", "FGF2", "BMP2+FGF2"]:
            params.base_frame = 400
        else:
            params.base_frame = 20

        weight_path_base = Path(f"./weight/2021-06-26/{params.dataset}")
        save_path_base = Path(f"./output/2021-06-26/{params.dataset}")

        # one image
        weight_path = weight_path_base.joinpath("step1_super_one")
        supervised_one(params, weight_path)

        output_path_base = save_path_base.joinpath("detection/step1_super_one")
        pred(params, weight_path, output_path_base)
        output_path_test = save_path_base.joinpath("detection/step1_super_one/test")
        pred_test(params, weight_path, output_path_test)
        save_path = save_path_base.joinpath(f"pseudo/step_1_pseudo_label")
        pseudo_label_gen(output_path_base, save_path, params)

        for iter in range(1, 5):
            weight_path = weight_path_base.joinpath(f"step_{iter}_pseudo_label")
            pseudo_train(params, save_path, weight_path)

            output_path_base = save_path_base.joinpath(f"detection/step_{iter}_pseudo_label")
            pred(params, weight_path, output_path_base)
            output_path_test = save_path_base.joinpath(f"detection/step_{iter}_pseudo_label/test")
            pred_test(params, weight_path, output_path_test)

            save_path = save_path_base.joinpath(f"pseudo/step_{iter + 1}_pseudo_label")
            pre_path = save_path_base.joinpath(f"pseudo/step_{iter}_pseudo_label")

            pseudo_label_gen_step2(params.dataset, pre_path, output_path_base, save_path, params)
