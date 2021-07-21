import numpy as np
from utils import heatmap_gen_per_cell
from matching import target_peaks_gen
from pathlib import Path
import cv2


if __name__ == "__main__":
    gt_paths = Path(
        "/home/kazuya/main/semisupervised_detection/image/c2c12/test/Control/9"
    ).glob("*.*")
    total_num = 0
    for frame, gt_path in enumerate(gt_paths):
        gt = cv2.imread(str(gt_path), 0)
        peaks = target_peaks_gen(gt)
        total_num += peaks.shape[0]
    print(total_num)
    print(total_num / frame)
    # pos = np.loadtxt(
    #     "/home/kazuya/main/semisupervised_detection/image/c2c12/train2/F0007.txt",
    #     skiprows=3,
    # )
    # img_paths = sorted(
    #     Path(
    #         "/home/kazuya/main/semisupervised_detection/image/c2c12/test/FGF2-2/img"
    #     ).glob("*.tif")
    # )

    # for img_path in img_paths:
    #     img = cv2.imread(str(img_path), -1)
    #     img = img / 4096 * 255
    #     cv2.imwrite(str(img_path), img.astype(np.uint8))

    # img = cv2.imread(str(img_paths[0]), -1)
    # pos = pos[pos[:, 2] == 499]
    # heatmap = heatmap_gen_per_cell(img.shape, pos[:, :2])
    # heatmap = np.max(heatmap, axis=0)
    # cv2.imwrite(
    #     "/home/kazuya/main/semisupervised_detection/image/c2c12/train2/FGF2/gt/00500.tif",
    #     heatmap.astype(np.uint8),
    # )
    print(1)
