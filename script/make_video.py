import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
import numpy as np


def make_vid(video_paths, save_path):
    img = cv2.imread(str(video_paths[0]), 0)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(save_path, fourcc, 5, (img.shape[0], img.shape[1]))

    for img_path in video_paths:
        img = cv2.imread(str(img_path))
        video.write(img)
    video.release()


def make_ov_vid(det_paths, pre_paths, save_path):
    img = cv2.imread(str(pre_paths[0]), 0)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(save_path, fourcc, 5, (img.shape[0], img.shape[1]))

    for det_path, pre_path in zip(det_paths, pre_paths):
        pre = cv2.imread(str(pre_path))
        det = Image.open(det_path).convert("RGB")
        det = np.array(det.resize(pre.shape[:2][::-1]), dtype=np.uint8)

        ad_im = cv2.addWeighted(det, 0.5, pre, 0.5, 0.1)
        video.write(ad_im)
    video.release()


DATASETS = {
    # 1: "BF-C2DL-HSC",
    # 2: "BF-C2DL-MuSC",
    3: "DIC-C2DH-HeLa",
    4: "PhC-C2DH-U373",
    5: "PhC-C2DL-PSC",
    6: "Fluo-C2DL-MSC",
    7: "Fluo-N2DH-GOWT1",
    8: "Fluo-N2DH-SIM+",
    9: "Fluo-N2DL-HeLa",
}
if __name__ == '__main__':
    for dataset in DATASETS.values():
        for seq in [1, 2]:
            for test_seq in [1, 2]:
                det_res = sorted(Path(
                    f"./output/select_pseudo/{dataset}/3-3/{seq:02d}-{test_seq:02d}/detect_result").glob(
                    "*.png"))
                pred_res = sorted(
                    Path(f"./output/detection/unsupervised_pred/{dataset}/{seq:02d}-{test_seq:02d}/pred").glob("*.tif"))

                save_name = f"./output/select_pseudo/{dataset}/3-3/{seq:02d}-{test_seq:02d}/det_ov_res.mp4"
                make_ov_vid(det_res, pred_res, save_name)

                # save_name = f"./output/select_pseudo/{dataset}/3-3/{seq:02d}-{test_seq:02d}/seledet_res.mp4"
                # make_vid(det_res, save_name)
