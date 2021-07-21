import numpy as np
import math
import pulp
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image

from .utils import local_maxim


def optimum(target, pred, dist_threshold):
    """
    :param target:target plots numpy [x,y]
    :param pred: pred plots numpy[x,y]
    :param dist_threshold: distance threshold
    :return: association result
    """
    r = 0.01
    # matrix to calculate
    c = np.zeros((0, pred.shape[0] + target.shape[0]))
    d = np.array([])
    associate_id = np.zeros((0, 2))

    # distance to GT position 
    for ii in range(int(target.shape[0])):
        dist = pred[:, 0:2] - np.tile(target[ii, 0:2], (pred.shape[0], 1))
        # calculate distance
        dist_lis = np.sqrt(np.sum(np.square(dist), axis=1))
        cc = np.where(dist_lis <= dist_threshold)[0]

        for j in cc:
            c1 = np.zeros((1, target.shape[0] + pred.shape[0]))
            c1[0, ii] = 1
            c1[0, target.shape[0] + j] = 1
            c = np.append(c, c1, axis=0)
            d = np.append(d, math.exp(-r * dist_lis[j]))
            associate_id = np.append(associate_id, [[ii, j]], axis=0)

    prob = pulp.LpProblem("review", pulp.LpMaximize)

    index = list(range(d.shape[0]))  # type:

    x_vars = pulp.LpVariable.dict("x", index, lowBound=0, upBound=1, cat="Integer")

    prob += sum([d[i] * x_vars[i] for i in range(d.shape[0])])

    for j in range(c.shape[1]):
        prob += sum([c[i, j] * x_vars[i] for i in range(d.shape[0])]) <= 1

    prob.solve()

    x_list = np.zeros(d.shape[0], dtype=int)
    for jj in range(d.shape[0]):
        x_list[jj] = int(x_vars[jj].value())
    associate_id = np.delete(associate_id, np.where(x_list == 0)[0], axis=0)
    return associate_id.astype(np.int)


def remove_outside_plot(matrix, associate_id, i, window_size, window_thresh=10):
    """
    delete peak that outside
    :param matrix:target matrix
    :param associate_id:optimize result
    :param i: 0 or 1 .0->target,1->pred
    :param window_size: window size
    :return: removed outside plots
    """
    # delete edge plot
    index = np.delete(np.arange(matrix.shape[0]), associate_id[:, i])
    if index.shape[0] != 0:
        a = np.where(
            (matrix[index][:, 0] < window_thresh) | (matrix[index][:, 0] > window_size[1] - window_thresh)
        )[0]
        b = np.where(
            (matrix[index][:, 1] < window_thresh) | (matrix[index][:, 1] > window_size[0] - window_thresh)
        )[0]
        delete_index = np.unique(np.append(a, b, axis=0))
        return (
            np.delete(matrix, index[delete_index], axis=0),
            np.delete(index, delete_index),
        )
    else:
        return matrix, np.array([])


def show_res(img, gt, res, no_detected_id, over_detection_id, path=None):
    plt.figure(figsize=(3, 3), dpi=500)
    plt.imshow(img, plt.cm.gray)
    plt.plot(gt[:, 0], gt[:, 1], "y3", label="gt_annotation", markersize=5)
    plt.plot(res[:, 0], res[:, 1], "g4", label="pred", markersize=5)
    if no_detected_id.shape[0] > 0:
        plt.plot(
            gt[no_detected_id][:, 0], gt[no_detected_id][:, 1], "b2", label="no_detected", markersize=5,

        )
    if over_detection_id.shape[0] > 0:
        plt.plot(
            res[over_detection_id][:, 0],
            res[over_detection_id][:, 1],
            "k1",
            label="over_detection",
            markersize=5,
        )
    plt.legend(bbox_to_anchor=(0, 1), loc="upper left", fontsize=4, ncol=4)
    plt.axis("off")
    plt.savefig(path, bbox_inches='tight', pad_inches=0, trasparent=True)
    plt.close()

def gt_id_gen():
    f_path = Path('./image/gt_id.txt')

    tree = ET.parse('./image/sequence18.xml')
    root = tree.getroot()
    annotations = []
    for i in root.findall(".//s"):
        annotations.append([int(float(i.get('i'))), int(float(i.get('x'))), int(float(i.get('y')))])
    j = 0
    annotations = np.array(annotations)
    with f_path.open(mode='w') as f:
        for i in range(600, 700):
            frame_per_annotations = annotations[annotations[:, 0] == (i - 1)]
            for annotation in frame_per_annotations:
                f.write('%d,%d,%d,%d\n' % (j, i, annotation[2], annotation[1]))
                j += 1


def associate(plot_size, gts, expert, df):
    res_paths = sorted(Path('/home/kazuya/ssd/detection/output/test18/MSELoss/%s/res' % plot_size).glob('*.tif'))
    ori_paths = sorted(Path('./image/originalTiff18').glob('*.tif'))
    paths = zip(res_paths, ori_paths)
    for i, path in enumerate(paths):
        # load image
        img = np.array(Image.open(str(path[0])))
        ori = np.array(Image.open(str(path[1])))
        # local local_maxima
        res = local_maxim(img, 100, 2)
        gt = gts[gts[:, 1] == i + 600][:, [3, 2, 0]]

        # associate
        associate_id = optimum(gt, res, 10).astype(int)

        # associated ID
        df.loc[gt[associate_id[:, 0]][:, 2], expert] = 1
    return df

