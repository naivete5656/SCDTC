from .load import CellImageLoad
import torch
import cv2
import numpy as np


class CellImageLoadPseudo(CellImageLoad):
    def __init__(self, ori_path, gt_path, bg_path, dataset, crop_size=(128, 128), crop=True):
        super().__init__(ori_path, gt_path, dataset, crop_size, crop)
        self.bg_paths = bg_path

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = self.load_img(img_name)

        gt_name = self.gt_paths[data_id]
        gt = cv2.imread(str(gt_name), 0)
        gt = gt / 255

        bg_name = self.bg_paths[data_id]
        if bg_name == 0:
            bg = np.ones_like(img)
        else:
            bg = cv2.imread(str(bg_name), 0)
            bg = bg / 255

        assert int(gt_name.name[:5]) == int(bg_name.name[:5]), "diffferent img name"

        if self.dataset in ["PhC-C2DL-PSC"]:
            bg = bg[40: gt.shape[0] - 40, 60: gt.shape[1] - 60]
            img = img[40: img.shape[0] - 40, 60: img.shape[1] - 60]
            gt = gt[40: gt.shape[0] - 40, 60: gt.shape[1] - 60]

        # img, gt, bg = self.data_augument(img, gt, bg)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        bg = torch.from_numpy(bg.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0), "bg": bg.unsqueeze(0)}
        return datas

