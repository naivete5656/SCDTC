from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale

from utils import heatmap_gen

CTC_DATASET = ["DIC-C2DH-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC"]


def mask2txt(img_paths, tra_paths, save_path):
    points = []
    for tra_path, img_path in zip(tra_paths, img_paths):
        tra = cv2.imread(str(tra_path), -1)
        img = cv2.imread(str(img_path), -1)
        for cell_id in np.unique(tra)[1:]:
            x, y = np.where(tra == cell_id)
            x_centroid = x.sum() / x.size
            y_centroid = y.sum() / y.size
            frame = int(tra_path.stem[-3:])
            points.append([frame, cell_id, y_centroid, x_centroid])

        points_np = np.array(points)
        points_np = points_np[points_np[:, 0] == frame]
        plt.figure(figsize=(3, 3), dpi=50)
        plt.imshow(img, plt.cm.gray)
        plt.plot(points_np[:, 2], points_np[:, 3], "rx")
        plt.axis("off")
        plt.savefig(str(save_path.joinpath(f"plot_img/{frame:03d}.png")), bbox_inches='tight', pad_inches=0,
                    trasparent=True)
        plt.close()
    np.savetxt(str(save_path.joinpath(f"gt_pos_{seq:02d}.txt")), points, fmt="%d")


def calculate_size(mask_paths):
    width_height = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), -1)
        for i in np.unique(mask)[1:]:
            mask_tmp = np.zeros_like(mask)
            mask_tmp[mask == i] = 255
            contours, hierarchy = cv2.findContours(mask_tmp.astype(np.uint8), 1, 2)
            rect = cv2.minAreaRect(contours[0])
            width_height.append(np.array(sorted(rect[1])))
    width_height = np.array(width_height)
    cell_size = width_height.mean(0)
    return cell_size


def rescale_dataset(img_paths, scale, save_path, seq):
    # rescale annotation
    gt_plot = np.loadtxt(str(save_path.joinpath(f"{seq:02d}/gt_pos_{seq:02d}.txt")))
    gt_plot[:, 2] = gt_plot[:, 2] * scale
    gt_plot[:, 3] = gt_plot[:, 3] * scale
    np.savetxt(str(save_path.joinpath(f"{seq:02d}/gt_pos_{seq:02d}_scaled.txt")), gt_plot)

    # rescale image
    for img_path in img_paths:
        img = cv2.imread(str(img_path), -1)
        image_rescaled = rescale(img, scale, anti_aliasing=False)
        image_rescaled = (image_rescaled - image_rescaled.min()) / (image_rescaled.max() - image_rescaled.min())
        cv2.imwrite(
            str(save_path.joinpath(f"{seq:02d}/img/{img_path.stem}.png")),
            (image_rescaled * 255).astype(np.uint8),
        )


def preprocessing_img(datasets, base_path):
    for dataset in datasets:
        cell_size_ave = np.zeros(2)
        for seq in [1, 2]:
            save_path = Path(f"./image/{dataset}/{seq:02d}")
            save_path.joinpath("plot_img").mkdir(parents=True, exist_ok=True)
            save_path.joinpath("img").mkdir(parents=True, exist_ok=True)

            img_paths = sorted(base_path.joinpath("{dataset}/{seq:02d}").glob("*.tif"))
            tra_paths = sorted(base_path.joinpath("{dataset}/{seq:02d}_GT/TRA").glob("*.tif"))
            mask_paths = sorted(base_path.joinpath("{dataset}/{seq:02d}_GT/SEG").glob("*.tif"))

            mask2txt(img_paths, tra_paths, save_path)
            cell_size_ave += calculate_size(mask_paths)

        cell_size_ave /= 2
        print(f"average: {cell_size_ave}")

        # Rescale image to fit Gaussian distribution with standard variance 9 (i. e., cell size about 3 * 9)
        # If cell size is smaller than 27 pixel, we rescale image to fit standard variance 6
        if cell_size_ave.max() < 27:
            # scale = 18 / cell_size_ave.min()
            scale = 1
            g_size = 6
        else:
            scale = 27 / cell_size_ave.min()
            g_size = 9

        for seq in [1, 2]:
            img_paths = sorted(base_path.joinpath("{dataset}/{seq:02d}").glob("*.tif"))
            rescale_dataset(img_paths, scale, save_path.parent, seq)

        for seq in [1, 2]:
            base_path = Path(f"./image/{dataset}/{seq:02d}")
            shape = cv2.imread(str(base_path.joinpath(f"img/t000.png")), 0).shape

            output_path = base_path.joinpath(f"gt")
            output_path.mkdir(parents=True, exist_ok=True)
            cell_positions = np.loadtxt(str(base_path.joinpath(f"gt_pos_{seq:02d}_scaled.txt")))
            heatmap_gen(shape, cell_positions[:, [0, 2, 3]], output_path, g_size=g_size, )
