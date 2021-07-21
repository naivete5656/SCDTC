import numpy as np
import cv2
from utils import heatmap_gen
from tracking import base_frame_track, track_from_first, select_pseudo, gen_mask_region


def tracking_detection_result(base_path, save_path, params):
    """
    Select continuously detected pred position longer than a threshold
    :param path: path of a detection result
    :param path: path of save directly
    :trac_len int: threshold for determining how many consecutive counts
    """
    save_path.joinpath("region").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("mask").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("img").mkdir(parents=True, exist_ok=True)

    img_paths = sorted(base_path.joinpath("img").glob("*.*"))
    lik_paths = sorted(base_path.joinpath("pred").glob("*.*"))

    # forward tracking from base frame
    iterator = enumerate(zip(img_paths[params.base_frame + 1:], lik_paths[params.base_frame + 1:]), params.base_frame)
    track_res_forward, non_assoc_forward, tracked_rate_f = base_frame_track(
        img_paths, lik_paths, params.base_frame, iterator, params, direction=1)

    # backward tracking from base frame
    iterator = reversed(list(enumerate(zip(img_paths[0:params.base_frame], lik_paths[0:params.base_frame]), 1)))
    track_res_backward, non_assoc_backward, tracked_rate_b = base_frame_track(
        img_paths, lik_paths, params.base_frame, iterator, params, direction=-1)

    # aggrigate tracked result
    track_from_base = np.append(track_res_forward, track_res_backward, axis=0)
    tracked_rate = np.append(tracked_rate_f, tracked_rate_b, axis=0)
    np.save(str(save_path.joinpath("tracked_rate")), tracked_rate)
    np.save(str(save_path.joinpath("tracked_rate_all.npy")), tracked_rate)
    np.save(str(save_path.joinpath("track_res")), track_from_base)

    # generate mask region
    iterator = range(track_from_base[:, 3].min(), track_from_base[:, 3].max() + 1)
    gen_mask_region(img_paths, lik_paths, iterator, track_from_base, non_assoc_backward, non_assoc_forward,
                    tracked_rate, save_path, params)

    # save img
    track_from_base = track_from_base.astype(np.int)
    for frame, img_path in enumerate(img_paths[track_from_base[:, 3].min(): track_from_base[:, 3].max() + 1],
                                     track_from_base[:, 3].min()):
        img = cv2.imread(str(img_path), -1)
        cv2.imwrite(str(save_path.joinpath(f"img/{frame:05d}.png")), img)


def tracking_detection_result_step2(base_path, pre_path, save_path, params):
    """
    Select continuously detected pred position longer than a threshold
    :param path: path of a detection result
    :param path: path of save directly
    :trac_len int: threshold for determining how many consecutive counts
    """
    save_path.joinpath("track_res").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("region").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("mask").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
    save_path.joinpath("associated_cell").mkdir(parents=True, exist_ok=True)
    img_paths = sorted(base_path.joinpath("img").glob("*.png"))
    lik_paths = sorted(base_path.joinpath("pred").glob("*.png"))

    # set start frame for backward and forward
    # start tracking from most correctly tracked frame
    tracked_rate_pre = np.load(str(pre_path.joinpath("tracked_rate.npy")))
    tracked_rate_pre_b = tracked_rate_pre[params.base_frame > tracked_rate_pre[:, 0]]
    if tracked_rate_pre_b[:, 1].shape[0] > 0:
        tracked_rate_b = max(tracked_rate_pre_b[:, 1])
        start_frame_b = int(min(tracked_rate_pre_b[tracked_rate_pre_b[:, 1] == tracked_rate_b][:, 0]))
    else:
        start_frame_b = None

    tracked_rate_pre_f = tracked_rate_pre[params.base_frame < tracked_rate_pre[:, 0]]
    if tracked_rate_pre_f[:, 1].shape[0] > 0:
        tracked_rate_f = max(tracked_rate_pre_f[:, 1])
        start_frame_f = int(max(tracked_rate_pre_f[tracked_rate_pre_f[:, 1] == tracked_rate_f][:, 0]))
    else:
        start_frame_f = None

    # forward tracking from start frame
    if (start_frame_f != len(img_paths)) & (start_frame_f is not None):
        iterator = enumerate(zip(img_paths[start_frame_f + 1:], lik_paths[start_frame_f + 1:]), start_frame_f)
        track_res_forward, non_assoc_forward, tracked_rate_f = base_frame_track(
            img_paths, lik_paths, start_frame_f, iterator, params, direction=1)
    else:
        track_res_forward = np.zeros((0, 4))
        non_assoc_forward = np.zeros((0, 4))
        tracked_rate_f = np.zeros((0, 2))

    if (start_frame_b != 0) & (start_frame_b is not None):
        iterator = reversed(list(enumerate(zip(img_paths[0:start_frame_b], lik_paths[0:start_frame_b]), 1)))
        track_res_backward, non_assoc_backward, tracked_rate_b = base_frame_track(
            img_paths, lik_paths, start_frame_b, iterator, params, direction=-1)
    else:
        track_res_backward = np.zeros((0, 4))
        non_assoc_backward = np.zeros((0, 4))
        tracked_rate_b = np.zeros((0, 2))
        start_frame_b = -1

    if (start_frame_b == -1) & (start_frame_f == None):
        track_res_final = np.zeros((0, 4))
        tracked_rate = np.zeros((0, 4))
    else:
        track_from_base = np.append(track_res_forward, track_res_backward, axis=0)
        tracked_rate = np.append(tracked_rate_b, tracked_rate_f, axis=0)
        # Track before
        iterator = enumerate(
            zip(img_paths[params.base_frame:start_frame_f - 1], lik_paths[params.base_frame:start_frame_f - 1]),
            params.base_frame + 1)
        track_res_for, max_cell_id = track_from_first(
            iterator, img_paths, lik_paths, base_frame=params.base_frame, direction=1, dist_th=params.dist_th)

        iterator = reversed(list(enumerate(
            zip(img_paths[start_frame_b + 1:params.base_frame], lik_paths[start_frame_b + 1:params.base_frame]),
            start_frame_b + 1)))
        track_res_back, new_cell_id = track_from_first(iterator, img_paths, lik_paths, base_frame=params.base_frame,
                                                       direction=-1, dist_th=params.dist_th, new_cell_id=max_cell_id)
        track_res_all = np.append(track_res_for, track_res_back[track_res_back[:, 3] != 20], axis=0)

        # add base frame result
        track_res_final = track_res_all
        track_res_final = np.append(track_from_base, track_res_final, axis=0)
        min_frame = int(max(track_res_final[:, 3].min(), params.base_frame - 100))
        max_frame = int(min(track_res_final[:, 3].max() + 1, params.base_frame + 100))
        iterator = range(min_frame, max_frame)
        gen_mask_region(img_paths, lik_paths, iterator, track_res_final, non_assoc_backward, non_assoc_forward,
                        tracked_rate, save_path, params)

        for frame, img_path in enumerate(img_paths[min_frame: max_frame], min_frame):
            img = cv2.imread(str(img_path), -1)
            cv2.imwrite(str(save_path.joinpath(f"img/{frame:05d}.png")), img)

    np.save(str(save_path.joinpath("track_res")), track_res_final)
    np.save(str(save_path.joinpath("tracked_rate")), tracked_rate)


def fg_pseudo_gen(base_path, track_res_path, gaus_size):
    img = cv2.imread(str(base_path.joinpath("img/00000.png")), 0)
    track_res = np.load(track_res_path)
    save_path = track_res_path.parent.joinpath("fg_pseudo")
    save_path.mkdir(parents=True, exist_ok=True)
    heatmap_gen(img.shape, track_res[:, [3, 0, 1]], save_path, gaus_size)


def back_mask_gen(base_path, save_path, params):
    track_res = np.load(save_path.joinpath("track_res.npy")).astype(np.int)
    pred_paths = sorted(base_path.joinpath("pred").glob("*.png"))
    if base_path.parents[1].name.split("_")[0] not in ["BMP2", "FGF2", "Control", "BMP2+FGF2"]:
        gt_paths = sorted(base_path.joinpath("gt").glob("*.png"))
    save_path_vis = save_path.joinpath("pseudo_label_vis")
    save_path = save_path.joinpath("bg_pseudo")
    fg_path = save_path.parent.joinpath("fg_pseudo")

    save_path.mkdir(parents=True, exist_ok=True)
    save_path_vis.mkdir(parents=True, exist_ok=True)

    # save bg mask base frame
    shape = cv2.imread(str(pred_paths[0]), 0).shape
    cv2.imwrite(str(save_path.joinpath(f"{params.base_frame:05d}.png")), np.full(shape, 255))

    for frame in range(track_res[:, 3].min(), track_res[:, 3].max() + 1):
        peaks = track_res[track_res[:, 3] == frame][:, :2]
        # load image
        if base_path.parents[1].name.split("_")[0] not in ["BMP2", "FGF2", "Control", "BMP2+FGF2"]:
            gt = cv2.imread(str(gt_paths[frame]), 1)
        fg = cv2.imread(str(save_path.parent.joinpath(f"fg_pseudo/{frame:05d}.png")), 1)
        ignore_mask = cv2.imread(str(save_path.parent.joinpath(f"mask/{frame:05d}.png")), 0)

        if (ignore_mask == 0).sum() == 0:
            cv2.imwrite(str(save_path.joinpath(f"{frame:05d}.png")), ignore_mask)
            fg = cv2.imread(str(save_path.parent.joinpath((f"fg_pseudo/{frame:05d}.png"))), 1)
            cv2.imwrite(str(save_path_vis.joinpath(f"{frame:05d}_fg.png")), fg)
            if base_path.parents[1].name.split("_")[0] not in ["BMP2", "FGF2", "Control", "BMP2+FGF2"]:
                cv2.imwrite(str(save_path_vis.joinpath(f"{frame:05d}_gt.png")), gt)
        else:
            # Determine the region of each peak. if the circle overlapped, we assign the pixel to nearest the peak.
            region = cv2.imread(str(save_path.parent.joinpath((f"region/{frame:05d}.png"))), 0)
            region_id_list = []
            for y, x in peaks.astype(np.int):
                region_id_list.append(region[x, y])
            region_new = np.zeros_like(region)
            for region_id in region_id_list:
                region_new[region == region_id] = 255

            mask = np.zeros_like(region, dtype=np.uint8)
            mask[ignore_mask < 255] = 0
            mask[ignore_mask == 255] = 255
            mask[region_new > 0] = 255
            cv2.imwrite(str(save_path.joinpath(f"{frame:05d}.png")), mask)

            fg[:, :, 2][mask < 255] = 255
            if base_path.parents[1].name.split("_")[0] not in ["BMP2", "FGF2", "Control", "BMP2+FGF2"]:
                gt[:, :, 2][mask < 255] = 255
            cv2.imwrite(str(save_path_vis.joinpath(f"{frame:05d}_fg.png")), fg)
            if base_path.parents[1].name.split("_")[0] not in ["BMP2", "FGF2", "Control", "BMP2+FGF2"]:
                cv2.imwrite(str(save_path_vis.joinpath(f"{frame:05d}_gt.png")), gt)


GAUS_SIZE = {
    "DIC-C2DH-HeLa": 9,
    "PhC-C2DH-U373": 9,
    "PhC-C2DL-PSC": 6,
    "BMP2": 9,
    "Control": 9,
    "FGF2": 9,
    "BMP2+FGF2": 9,
}

CTC_dataset = ["DIC-C2DH-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC"]


def pseudo_label_gen(base_path, save_path_base, params):
    if params.dataset in CTC_dataset:
        seq_list = ["01/", "02/"]
    else:
        seq_list = [""]
    for seq in seq_list:
        save_path = save_path_base.joinpath(f"{seq}")
        pred_path = base_path.joinpath(f"{seq}")
        global g_sigma
        g_sigma = GAUS_SIZE[params.dataset]

        # select accurate result
        tracking_detection_result(pred_path, save_path, params)

        # # gen fg pseudo mask
        fg_pseudo_gen(pred_path, save_path.joinpath("track_res.npy"), GAUS_SIZE[params.dataset])

        # gen bg pseudo mask
        back_mask_gen(pred_path, save_path, params)


def pseudo_label_gen_step2(dataset, pre_path_base, base_path, save_path_base, params):
    if dataset in CTC_dataset:
        seq_list = ["01/", "02/"]
    else:
        seq_list = [""]
    for seq in seq_list:
        save_path = save_path_base.joinpath(f"{seq}")
        pred_path = base_path.joinpath(f"{seq}")
        pre_path = pre_path_base.joinpath(f"{seq}")

        global g_sigma
        g_sigma = GAUS_SIZE[dataset]

        # select accurate result
        tracking_detection_result_step2(pred_path, pre_path, save_path, params)

        # # gen fg pseudo mask
        fg_pseudo_gen(pred_path, save_path.joinpath("track_res.npy"), GAUS_SIZE[dataset])

        # gen bg pseudo mask
        back_mask_gen(pred_path, save_path, params)
