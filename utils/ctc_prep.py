from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from skimage.transform import rescale, resize, downscale_local_mean
import re
from utils import heatmap_gen

def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(
        img, (pad_size, pad_size), "constant"
    )  # zero padding(これしないと正規化後、画像端付近の尤度だけ明るくなる)
    img_t = cv2.GaussianBlur(
        img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma
    )  # filter gaussian(適宜パラメータ調整)
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t


def cal_size(mask_paths, original_path):
    width_height = []
    img_maxs = []
    for mask_path, ori_path in zip(mask_paths, original_path):
        img = cv2.imread(str(ori_path), -1)
        img_maxs.append(img.max())

        mask = cv2.imread(str(mask_path), -1)
        for i in range(1, mask.max() + 1):
            mask_tmp = np.zeros_like(mask)
            mask_tmp[mask == i] = 255
            if mask_tmp.max() > 0:
                contours, hierarchy = cv2.findContours(mask_tmp.astype(np.uint8), 1, 2)
                rect = cv2.minAreaRect(contours[0])
                width_height.append(np.array(sorted(rect[1])))
    width_height = np.array(width_height)
    cell_size = width_height.mean(0)

    img_max = np.array(img_maxs).max()
    print(dataset, cell_size, img_max)


def position_generate(dataset, original_img_path, tracking_gt_path, save_path):
    if dataset == "PhC-C2DL-PSC":
        original_img_path = original_img_path[150:250]

    points = []
    for frame, (tra_path, ori_path) in enumerate(
            zip(tracking_gt_path, original_img_path)
    ):
        tra_gt = cv2.imread(str(tra_path), -1)
        ori = cv2.imread(str(ori_path))
        ids = np.unique(tra_gt)
        ids = ids[ids != 0]
        # plt.imshow(img), plt.show()
        if dataset == "PhC-C2DL-PSC":
            frame = 150 + frame
        for cell_id in ids:
            x, y = np.where(tra_gt == cell_id)
            x_centor = x.sum() / x.size
            y_centor = y.sum() / y.size
            points.append([frame, cell_id, y_centor, x_centor])

        points2 = np.array(points)
        points2 = points2[points2[:, 0] == frame]
        plt.imshow(ori), plt.plot(points2[:, 2], points2[:, 3], "rx"), plt.savefig(
            "test.png"
        ), plt.close()
    np.savetxt(str(save_path), points, delimiter=",", fmt="%d")


def mask2pos():
    for dataset in datasets.values():
        for sequence in [1, 2]:
            tracking_gt_path = sorted(
                Path(
                    f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/TRA"
                ).glob("*.tif")
            )

            original_img_path = sorted(
                Path(
                    f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}"
                ).glob(f"*.tif")
            )
            save_path = Path(
                f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/gt_plots_{sequence:02d}.txt"
            )

            position_generate(dataset, original_img_path, tracking_gt_path, save_path)


def scale_original_image():
    scaled_dataset = []
    gaus_size_dataset = {}

    root_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/test")
    # args = parse_args()
    for dataset in datasets.values():
        img_paths = sorted(root_path.joinpath(f"{dataset}/01").glob("*.tif"))
        img = cv2.imread(str(img_paths[0]), -1)
        scale_9 = 27 / (
            (np.array(SCALE[dataset][1][0] + np.array(SCALE[dataset][2][0])) / 2)
        )
        scale_6 = 16 / (
            (np.array(SCALE[dataset][1][0] + np.array(SCALE[dataset][2][0])) / 2)
        )
        gaus_size_index = np.argmin([abs(scale_6 - 1), abs(scale_9 - 1)])
        gaus_size = GAUS_SIZE[gaus_size_index]
        scale = [scale_6, scale_9][gaus_size_index]
        gaus_size_dataset.setdefault(dataset, gaus_size)
        if (scale < 0.8) or (scale > 1.2):
            scaled_dataset.append([dataset, scale])

        for sequence in [1, 2]:
            ori_paths = sorted(root_path.joinpath(f"{dataset}/{sequence:02d}").glob(f"*.tif"))
            save_scale_path = root_path.joinpath(f"{dataset}/{sequence:02d}-S")

            if (scale < 0.8) or (scale > 1.2):
                # gt_plot = np.loadtxt(
                #     f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/gt_plots_{sequence:02d}.txt",
                #     delimiter=",",
                # )
                # gt_plot[:, 2] = gt_plot[:, 2] * scale
                # gt_plot[:, 3] = gt_plot[:, 3] * scale
                #
                # np.savetxt(
                #     f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/gt_plots_{sequence:02d}_scale.txt",
                #     gt_plot,
                #     delimiter=",",
                # )

                save_scale_path.mkdir(parents=True, exist_ok=True)

                for ori_path in ori_paths:
                    img = cv2.imread(str(ori_path), -1)
                    size = img.shape

                    image_rescaled = rescale(img, scale, anti_aliasing=False)
                    image_rescaled = (image_rescaled - image_rescaled.min()) / (image_rescaled.max() - image_rescaled.min())
                    cv2.imwrite(
                        str(save_scale_path.joinpath(f"{ori_path.stem}.png")),
                        (image_rescaled * 255).astype(np.uint8),
                    )
    # import json

    # json.dump("/home/kazuya/dataset/Cell_tracking_challenge/gaus_size.txt")

    # with Path("/home/kazuya/dataset/Cell_tracking_challenge/scaled_dataset.txt").open(
    #         "w"
    # ) as f:
    #     for dataset in scaled_dataset:
    #         f.write(f"{dataset[0]}, {dataset[1]}" + "\n")

datasets = {
    # 1: "BF-C2DL-HSC",
    # 2: "BF-C2DL-MuSC",
    3: "DIC-C2DH-HeLa",
    # 4: "Fluo-C2DL-MSC",
    # 5: "Fluo-N2DH-GOWT1",
    # 6: "Fluo-N2DH-SIM+",
    # 7: "Fluo-N2DL-HeLa",
    # 8: "PhC-C2DH-U373",
    # 9: "PhC-C2DL-PSC"
}

SCALE = {
    "BF-C2DL-HSC": {1: [17.0793067, 24.51958351], 2: [17.29787348, 20.67317243]},
    "BF-C2DL-MuSC": {1: [27.11272188, 63.34942082], 2: [33.20797378, 104.06867167]},
    "DIC-C2DH-HeLa": {1: [93.85932307, 141.89484982], 2: [98.84045712, 149.01030849]},
    "Fluo-C2DL-MSC": {1: [59.30023488, 206.03586401], 2: [90.83024177, 366.07400345]},
    "Fluo-N2DH-GOWT1": {1: [54.63117686, 63.246417], 2: [56.95861561, 70.06671024]},
    "Fluo-N2DH-SIM+": {1: [41.23117971, 51.16462622], 2: [36.67656445, 47.92538745]},
    "Fluo-N2DL-HeLa": {1: [19.73457727, 26.72722964], 2: [22.39316751, 28.02623763]},
    "PhC-C2DH-U373": {1: [65.09892616, 91.22308933], 2: [57.50729906, 82.49417877]},
    "PhC-C2DL-PSC": {1: [9.11820082, 24.65092418], 2: [9.48466116, 22.19748413]},
}

GAUS_SIZE = [6, 9]
GAUS_SIZE1 = {
    "BF-C2DL-HSC": 6,
    "BF-C2DL-MuSC": 9,
    "DIC-C2DH-HeLa": 15,
    "Fluo-C2DL-MSC": 9,
    "Fluo-N2DH-GOWT1": 9,
    "Fluo-N2DH-SIM+": 9,
    "Fluo-N2DL-HeLa": 6,
    "PhC-C2DH-U373": 9,
    "PhC-C2DL-PSC": 3,
}

scale_size = [18, 27]


if __name__ == "__main__":
    # for dataset in datasets.values():
    #     base_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}")
    #     for seq in [1, 2]:
    #         shape = cv2.imread(str(base_path.joinpath(f"{seq:02d}/t000.tif")), 0).shape
    #
    #         output_path = base_path.joinpath(f"{seq:02d}_GT/LIK2")
    #         output_path.mkdir(parents=True, exist_ok=True)
    #         cell_positions = np.loadtxt(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/gt_plots_{seq:02d}.txt", delimiter=",")
    #         heatmap_gen(shape, cell_positions[:, [0, 3, 2]], GAUS_SIZE1[dataset], output_path)
    scale_original_image()

    # args = parse_args()
    # with Path("/home/kazuya/dataset/Cell_tracking_challenge/scaled_dataset.txt").open(
    #         "r"
    # ) as f:
    #     scaled_dataset = f.read()
    # scaled_dataset = scaled_dataset.split("\n")
    # scaled_dataset.pop(-1)
    #
    # with Path("/home/kazuya/dataset/Cell_tracking_challenge/gaus_size.txt").open(
    #         "r"
    # ) as f:
    #     gaus_size = f.read()
    # gaus_size = re.split("[,\n]", gaus_size)
    #
    # for dataset in datasets.values():
    #     for sequence in [1, 2]:
    #
    #         if dataset in scaled_dataset:
    #             args.input_path = f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/gt_plots_{sequence:02d}_scale.txt"
    #             args.output_path = Path(
    #                 f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/LIK-S"
    #             )
    #             original_img_path = sorted(
    #                 Path(
    #                     f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}-S"
    #                 ).glob(f"*.png")
    #             )
    #         else:
    #             args.input_path = f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/gt_plots_{sequence:02d}.txt"
    #             args.output_path = Path(
    #                 f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}_GT/LIK"
    #             )
    #             original_img_path = sorted(
    #                 Path(
    #                     f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{sequence:02d}"
    #                 ).glob(f"*.tif")
    #             )
    #
    #         args.output_path.mkdir(parents=True, exist_ok=True)
    #         args.g_size = GAUS_SIZE[dataset]
    #
    #         img = cv2.imread(str(original_img_path[0]), 0)
    #         args.height, args.width = img.shape
    #         likelyhood_map_gen(args)
