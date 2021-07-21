import random
import torch
from scipy.ndimage.interpolation import rotate
import numpy as np
import cv2


class CellImageLoad(object):
    def __init__(self, ori_path, gt_path, dataset, crop_size=(256, 256), crop=True):
        self.ori_paths = ori_path
        self.gt_paths = gt_path
        self.dataset = dataset
        self.crop_size = crop_size
        self.crop = crop

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = self.load_img(img_name)

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gt = gt / 255

        # PhC-C2DL-PSC do not have annotation on the edge of image. Cut edge of image
        if self.dataset in ["PhC-C2DL-PSC"]:
            img = img[40: gt.shape[0] - 40, 40: gt.shape[1] - 40]
            gt = gt[40: gt.shape[0] - 40, 40: gt.shape[1] - 40]

        # img, gt = self.data_augument(img, gt)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0)}
        return datas

    def load_img(self, img_name):
        img = cv2.imread(str(img_name), -1)
        img = img / 255
        return img

    # def data_augument(self, img, gt):
    #     if self.crop:
    #         # random crop
    #         top, bottom, left, right = self.random_crop_param(img.shape[:2])
    #         img = img[top:bottom, left:right]
    #         gt = gt[top:bottom, left:right]
    #
    #         # random rotation
    #         rand_value = np.random.randint(0, 4)
    #         img = rotate(img, 90 * rand_value, mode="nearest")
    #         gt = rotate(gt, 90 * rand_value)
    #     # Brightness
    #     pix_add = random.uniform(-0.1, 0.1)
    #     img = img + pix_add
    #     img = (img - img.min()) / (1 + pix_add - img.min())
    #     return img, gt

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right


class CellImageLoadTest(CellImageLoad):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __getitem__(self, data_id):
        img_name = self.img_paths[data_id]
        img = self.load_img(img_name)
        img = torch.from_numpy(img.astype(np.float32))
        img = img.unsqueeze(0)
        datas = {"image": img}
        return datas
