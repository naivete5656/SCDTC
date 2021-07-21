import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from utils import gaus_filter
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="txt path",
        default="/home/kazuya/main/semisupervised_detection/image/c2c12/test/BMP2+FGF2/plot.txt",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="/home/kazuya/main/semisupervised_detection/image/c2c12/test/BMP2+FGF2/9",
        type=str,
    )
    parser.add_argument(
        "-w", "--width", dest="width", help="image width", default=1392, type=int
    )
    parser.add_argument(
        "-he", "--height", dest="height", help="height", default=1040, type=int
    )
    parser.add_argument(
        "-g",
        "--gaussian_variance",
        dest="g_size",
        help="gaussian variance",
        default=9,
        type=int,
    )

    args = parser.parse_args()
    return args


def like_map_gen(args):
    args.output_path.mkdir(parents=True, exist_ok=True)
    # load txt file
    # cell_positions = np.loadtxt(args.input_path, delimiter=",", skiprows=1)
    cell_positions = np.loadtxt(args.input_path)
    black = np.zeros((args.height, args.width))

    # 1013 - number of frame
    for i in range(600, int(cell_positions[:, 0].max())):
        # likelihood map of one input
        result = black.copy()
        cells = cell_positions[cell_positions[:, 2] == i]
        for x, y, _, _ in cells:
            img_t = black.copy()  # likelihood map of one cell
            img_t[int(y)][int(x)] = 255  # plot a white dot
            img_t = gaus_filter(img_t, 301, args.g_size)
            result = np.maximum(result, img_t)  # compare result with gaussian_img
        #  normalization
        result = 255 * result / result.max()
        result = result.astype("uint8")
        cv2.imwrite(str(args.output_path / Path("%05d.tif" % i)), result)
        print(i + 1)
    print("finish")


if __name__ == "__main__":
    # args = parse_args()
    # args.input_path = f"/home/kazuya/dataset/riken_cut/B2_1/centroidof100frames.txt"
    # args.output_path = "/home/kazuya/dataset/riken_cut/B2_1/lik_6"
    # args.output_path = Path(args.output_path)
    # args.width = 1272
    # args.height = 952
    import cv2

    img = cv2.imread(
        "/home/kazuya/main/semisupervised_detection/image/c2c12/test/BMP2+FGF2/img/exp1_F0013-00600.tif",
        0,
    )
    from utils import heatmap_gen

    positions = np.loadtxt(
        "/home/kazuya/main/semisupervised_detection/image/c2c12/test/F0006-300-400.txt",
        skiprows=3,
    )
    save_path = Path(
        "/home/kazuya/main/semisupervised_detection/image/c2c12/test/FGF2-test/9"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    heatmap_gen(
        img.shape,
        positions[:, [2, 0, 1]],
        save_path,
    )
    # like_map_gen(args)
    # from pathlib import Path

    # positions = np.loadtxt("/home/kazuya/dataset/C2C12P7/sequence/semi-supervised/train/F0010.txt", skiprows=3)
    # imgs_path = sorted(Path("/home/kazuya/dataset/C2C12P7/sequence/semi-supervised/train/BMP2/img").glob("*.tif"))
    # for img_path in imgs_path:
    #     img = cv2.imread(str(img_path), -1)
    #     img = img / 4095 * 255
    #     cv2.imwrite(str(img_path), img.astype(np.uint8))

    # cell_pos_frame = positions[positions[:, 2] == 100]
    # from utils import heatmap_gen_per_cell
    # heatmap_gen_per_cell(img.shape, positions[:, :2])

    # import matplotlib.pyplot as plt
    # plt.imshow(img), plt.plot(cell_pos_frame[:, 0], cell_pos_frame[:, 1], "rx"), plt.show()
    # 1
