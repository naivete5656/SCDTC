import numpy as np
import cv2
from pathlib import Path
import random
import os
import torch
from skimage.feature import peak_local_max


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def gather_path(train_paths, mode, extension):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob(extension)))
    return ori_paths


def gaus_filter(img, kernel_size, sigma):
    pad_size = int(kernel_size - 1 / 2)
    img_t = np.pad(img, (pad_size, pad_size), "constant")  # zero padding
    img_t = cv2.GaussianBlur(
        img_t, ksize=(kernel_size, kernel_size), sigmaX=sigma
    )  # gaussian filter
    img_t = img_t[pad_size:-pad_size, pad_size:-pad_size]  # remove padding
    return img_t


def local_maxim(img, threshold, dist):
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0)
    return data


def target_peaks_gen(img):
    gt_plot = np.zeros((0, 2))
    x, y = np.where(img == 255)
    for j in range(x.shape[0]):
        gt_plot = np.append(gt_plot, [[y[j], x[j]]], axis=0)
    return gt_plot


def heatmap_gen(shape, cell_positions, save_path, g_size=9, bg_th=0):
    pad_size = g_size * 5 + abs((g_size * 5) % 2 - 1)
    black = np.zeros((shape[0], shape[1]))
    heatmap_basis = np.zeros((pad_size, pad_size))
    half_size = int((pad_size) / 2)
    heatmap_basis[half_size + 1, half_size + 1] = 255
    heatmap_basis = gaus_filter(heatmap_basis, 201, g_size)
    # 1013 - number of frame
    for frame in range(
        int(cell_positions[:, 0].min()), int(cell_positions[:, 0].max()) + 1
    ):
        # likelihood map of one input
        result = black.copy()
        cells = cell_positions[cell_positions[:, 0] == frame]
        for _, y, x in cells:
            img_t = np.zeros(
                (shape[0] + pad_size - 1, shape[1] + pad_size - 1)
            )  # likelihood map of one cell
            y_min, y_max = int(y), int(y + (pad_size))
            x_min, x_max = int(x), int(x + (pad_size))
            img_t[x_min:x_max, y_min:y_max] = heatmap_basis
            result = np.maximum(
                result,
                img_t[
                    half_size : abs(13 % 2 - 1) - half_size,
                    half_size : abs(13 % 2 - 1) - half_size,
                ],
            )
            #  normalization
        result = 255 * result / result.max()
        result = result.astype("uint8")
        result[result < bg_th] = 0
        cv2.imwrite(str(save_path / Path("%05d.png" % frame)), result)
        print(frame + 1)
    print("finish")


def heatmap_gen_per_cell(shape, cell_positions, g_size=9):
    heatmap_basis = np.zeros((g_size * 11, g_size * 11))
    half_size = int((g_size * 11) / 2)
    heatmap_basis[half_size + 1, half_size + 1] = 255
    heatmap_basis = gaus_filter(heatmap_basis, 101, g_size)
    results = [np.zeros((shape[0], shape[1]))]
    for y, x in cell_positions:
        img_t = np.zeros(
            (shape[0] + g_size * 11 - 1, shape[1] + g_size * 11 - 1)
        )  # likelihood map of one cell
        y_min, y_max = int(y), int(y + (g_size * 11))
        x_min, x_max = int(x), int(x + (g_size * 11))
        img_t[x_min:x_max, y_min:y_max] = heatmap_basis
        results.append(img_t[half_size:-half_size, half_size:-half_size])
    results = np.array(results)
    results = results / results.max() * 255
    return results

