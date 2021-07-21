import numpy as np
import cv2
from utils import local_maxim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from utils import (
    local_maxim,
    heatmap_gen_per_cell,
)
from skimage.draw import circle


def initalize(paths, track_res, base_frame, new_cell_id=0):
    det_res = det_cell(paths).astype(np.int)
    for idx, (x, y) in enumerate(det_res):
        track_res = np.append(track_res, [[x, y, idx, base_frame]], axis=0)
        new_cell_id += 1
    return track_res, new_cell_id


def det_cell(path, detect_th=125, peak_dist=5, debug=False):
    pred = cv2.imread(str(path[1]), 0)
    pred_plot = local_maxim(pred, detect_th, peak_dist)
    if debug:
        img = cv2.imread(str(path[0]), 0)
        plt.imshow(img), plt.plot(pred_plot[:, 0], pred_plot[:, 1], "rx"), plt.show()
    return pred_plot


def gen_mask_region(
        img_paths,
        lik_paths,
        iterator,
        track_res_final,
        non_assoc_backward,
        non_assoc_forward,
        track_rate,
        save_path,
        params,
):
    for frame in iterator:
        paths = [img_paths[params.base_frame], lik_paths[params.base_frame]]
        save_path_each = save_path.joinpath(f"region/{frame:05d}.png")
        save_path_each_mask = save_path.joinpath(f"mask/{frame:05d}.png")
        if (params.base_frame >= frame):
            if (params.base_frame == frame) or (
                    (frame in track_rate[:, 0]) and (track_rate[track_rate[:, 0] == frame][0][1] == 1)):
                img = cv2.imread(str(paths[0]))
                mask = np.full_like(img, 255)
                cv2.imwrite(str(save_path_each_mask), mask)
            else:
                mask_region_gen_backward(frame, track_res_final, non_assoc_backward, paths, save_path_each,
                                         save_path_each_mask, base_frame=params.base_frame)
        else:
            mask_region_gen_forward(frame, track_res_final, non_assoc_forward, paths, save_path_each,
                                    save_path_each_mask, base_frame=params.base_frame)


def mask_region_gen_backward(
        frame,
        track_res_final,
        non_assoc_base,
        paths,
        save_path_each,
        save_path_each_mask,
        base_frame=0,
        g_sigma=9
):
    # track res [x, y, id, frame]
    img = cv2.imread(str(paths[0]))
    track_res_frame = track_res_final[track_res_final[:, 3] == frame]

    points = track_res_frame[:, :2]

    non_assoc_cell_list = non_assoc_base[(non_assoc_base[:, 3] >= frame) & (non_assoc_base[:, 3] < base_frame)]
    points = np.vstack((points, non_assoc_cell_list[:, :2]))

    heatmaps = heatmap_gen_per_cell(img.shape[:2], points, g_size=g_sigma)
    heatmaps_new = np.zeros_like(heatmaps)
    for idx, point in enumerate(points, 1):
        rr, cc = circle(point[1], point[0], int(g_sigma * 3 / 2), img.shape[:2])
        heatmaps_new[idx][rr, cc] = heatmaps[idx][rr, cc]
    region = np.argmax(heatmaps_new, axis=0)
    cv2.imwrite(str(save_path_each), region)

    mask = np.full((img.shape[:2]), 255)
    for non_assoc_cell in non_assoc_cell_list:
        rr, cc = circle(non_assoc_cell[1], non_assoc_cell[0], int(g_sigma * 2 + non_assoc_cell[3] - frame),
                        img.shape[:2])
        mask[rr, cc] = 0
    cv2.imwrite(str(save_path_each_mask), mask)


def mask_region_gen_forward(
        frame,
        track_res_final,
        non_assoc,
        paths,
        save_path_each,
        save_path_each_mask,
        base_frame=0,
        g_sigma=9
):
    # track res [x, y, id, frame]
    img = cv2.imread(str(paths[0]))
    track_res_frame = track_res_final[track_res_final[:, 3] == frame]

    points = track_res_frame[:, :2]

    non_assoc_list = non_assoc[non_assoc[:, 2] == 0]
    non_assoc_list = non_assoc_list[(non_assoc_list[:, 3] > base_frame) & (non_assoc_list[:, 3] <= frame)]
    points = np.vstack((points, non_assoc_list[:, :2]))

    overdet_list = non_assoc[non_assoc[:, 2] == 1]
    overdet_list = overdet_list[(overdet_list[:, 3] >= frame - 2) & (overdet_list[:, 3] <= frame + 2)]
    points = np.vstack((points, overdet_list[:, :2]))

    heatmaps = heatmap_gen_per_cell(img.shape[:2], points, g_size=g_sigma)
    heatmaps_new = np.zeros_like(heatmaps)
    for idx, point in enumerate(points, 1):
        rr, cc = circle(point[1], point[0], int(g_sigma * 3 / 2), img.shape[:2])
        heatmaps_new[idx][rr, cc] = heatmaps[idx][rr, cc]
    region = np.argmax(heatmaps_new, axis=0)
    cv2.imwrite(str(save_path_each), region)

    mask = np.full((img.shape[:2]), 255)
    for non_assoc_cell in non_assoc_list:
        rr, cc = circle(non_assoc_cell[1], non_assoc_cell[0], int(g_sigma * 2 + non_assoc_cell[3]), img.shape[:2])
        mask[rr, cc] = 0
    for overdet_cell in overdet_list:
        rr, cc = circle(overdet_cell[1], overdet_cell[0], int(g_sigma * 2), img.shape[:2])
        mask[rr, cc] = 0
    cv2.imwrite(str(save_path_each_mask), mask)


def select_pseudo(track_res, trac_len_th=3):
    track_res_new = np.zeros((0, 4))
    short_term_cells = np.zeros((0, 4))
    for id in np.unique(track_res[:, 2]):
        if sum(track_res[:, 2] == id) > trac_len_th:
            track_res_new = np.append(
                track_res_new, track_res[track_res[:, 2] == id], axis=0
            )
        else:
            short_term_cells = np.append(
                short_term_cells, track_res[track_res[:, 2] == id], axis=0
            )
    return track_res_new, short_term_cells
