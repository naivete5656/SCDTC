from utils import optimum
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .tracking_utils import det_cell, initalize


def update_track_base(track_res, non_assoc, det, frame, params, path, direction=1):
    # track_res_np = np.array(track_res)
    pre_det = track_res[track_res[:, 3] == frame - direction]
    assoc_ids = optimum(pre_det, det, params.dist_th)

    # record tracked cell
    for assoc_id in assoc_ids:
        x = det[int(assoc_id[1])][0]
        y = det[int(assoc_id[1])][1]
        id = pre_det[int(assoc_id[0])][2]
        track_res = np.append(track_res, [[x, y, id, frame]], axis=0)

    if direction == 1:
        # append not associated cell to non_assoc array
        # associate cell in non associate cell
        non_assoc_det = det[np.setdiff1d(range(det.shape[0]), assoc_ids[:, 1])]

        # if the cell not associate any cell, newly add cell
        for idx in np.setdiff1d(range(non_assoc_det.shape[0]), non_assoc_det[:, 1]):
            non_assoc = np.append(non_assoc, [[non_assoc_det[idx][0], non_assoc_det[idx][1], 1, frame]], axis=0)

    # unassociate cell of base frame
    for idx in np.setdiff1d(range(pre_det.shape[0]), assoc_ids[:, 0]):
        non_assoc = np.append(non_assoc, [[pre_det[idx][0], pre_det[idx][1], 0, frame]], axis=0)
    return track_res, non_assoc, assoc_ids.shape[0]


def base_frame_track(img_paths, lik_paths, base_frame, iterator, params, direction):
    track_res = np.zeros((0, 4))  # [x, y, id, frame]
    non_assoc_res = np.zeros((0, 4))  # [x, y, id, frame]
    tracked_rate = np.zeros((0, 2))
    track_res, num_cell = initalize([img_paths[base_frame], lik_paths[base_frame]], track_res, base_frame)
    for frame, paths in iterator:
        det_res = det_cell(paths).astype(np.int)
        track_res, non_assoc_res, num_assoc_cell = update_track_base(track_res, non_assoc_res, det_res,
                                                                     frame + direction, params, paths,
                                                                     direction=direction)
        tracked_rate = np.append(tracked_rate, [[frame + direction, num_assoc_cell / num_cell]], axis=0)
        if num_cell * params.rate_of_failure > num_assoc_cell:
            break
    return track_res.astype(np.int), non_assoc_res.astype(np.int), np.array(tracked_rate)
