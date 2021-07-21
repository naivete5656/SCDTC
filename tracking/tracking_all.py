import numpy as np
from utils import optimum
from .tracking_utils import initalize, det_cell

def update_track(track_res, det, frame, new_cell_id, direction=1, dist_th=8, debug=False):
    # track_res_np = np.array(track_res)
    pre_det = track_res[track_res[:, 3] == frame - direction]
    assoc_ids = optimum(pre_det, det, dist_th)

    for assoc_id in assoc_ids:
        x = det[int(assoc_id[1])][0]
        y = det[int(assoc_id[1])][1]
        id = pre_det[int(assoc_id[0])][2]
        track_res = np.append(track_res, [[x, y, id, frame]], axis=0)

    for idx in set(range(det.shape[0])) - set(assoc_ids[:, 1]):
        track_res = np.append(
            track_res, [[det[idx][0], det[idx][1], new_cell_id, frame]], axis=0
        )
        new_cell_id += 1

    return track_res, new_cell_id


def track_from_first(iterator, img_paths, lik_paths, base_frame, direction=1, dist_th=8, new_cell_id=0):

    track_res = np.zeros((0, 4))  # [x, y, id, frame]
    track_res, new_cell_id = initalize(
        [img_paths[base_frame], lik_paths[base_frame]], track_res, base_frame, new_cell_id
    )

    for frame, paths in iterator:
        det_res = det_cell(paths).astype(np.int)
        track_res, new_cell_id = update_track(
            track_res, det_res, frame, new_cell_id, dist_th=dist_th, direction=direction
        )
    return track_res, new_cell_id