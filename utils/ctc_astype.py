from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import math


MODES = {
         # 1: "C2C12",
         # 2: "MoNuSeg",
         # 3: "TNBC",
         # 4: "DIC-C2DH-HeLa",
         # 5: "Fluo-C2DL-MSC",
        6: "Fluo-N2DH-GOWT1",
        #  7: "Fluo-N2DH-SIM+",
        #  8: "Fluo-N2DL-HeLa",
         9: "PhC-C2DH-U373",
         10: "PhC-C2DL-PSC", }

CTC_config = {"DIC-C2DH-HeLa": "s_scale_norm",
              "Fluo-C2DL-MSC": "l_scale_norm",
              "Fluo-N2DH-GOWT1": "s_scale_norm",
              "Fluo-N2DH-SIM+": "ori_norm",
              "Fluo-N2DL-HeLa": "s_scale",
              "PhC-C2DH-U373": "l_scale",
              "PhC-C2DL-PSC": "l_scale"}


if __name__=='__main__':
    dataset = MODES[6]
    pred_paths = sorted(Path(f"/home/kazuya/dataset/journal-output/graphcut/{dataset}/0.01/labelresults").glob("*.tif"))
    save_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/01_RES")
    save_path.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/01/t000.tif", 0)
    scale_img = np.load(f"/home/kazuya/dataset/Cell_tracking_challenge/detection/{dataset}/sequ01/s_scale_norm/t000.npy")
    size = scale_img.shape
    w = size[0]
    w_size = math.ceil(w / 64) * 64
    w_pad_size = math.ceil((w_size - w) / 2)
    h = size[1]
    h_size = math.ceil(h / 64) * 64
    h_pad_size = math.ceil((h_size - h) / 2)
    for t, pred_path in enumerate(pred_paths):
        pred = cv2.imread(str(pred_path), 0)
        pred = pred[w_pad_size: pred.shape[0] - w_pad_size, h_pad_size: pred.shape[1] - h_pad_size]
        pred = Image.fromarray(np.uint8(pred))
        pred = np.asarray(pred.resize(img.shape))
        cv2.imwrite(str(save_path.joinpath(f"mask{t:03d}.tif")), pred.astype(np.uint16))