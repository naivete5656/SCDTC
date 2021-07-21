from pathlib import Path
import pickle

MODES = {
        1: "Fluo-C2DL-MSC",
        2: "Fluo-N2DH-GOWT1",
        3: "Fluo-N2DH-SIM+",
        4: "Fluo-N2DL-HeLa",
        # 4: "BF-C2DL-HSC",
        # 5: "BF-C2DL-MuSC",
        # 6: "DIC-C2DH-HeLa",
        # 11: "PhC-C2DH-U373",
        # 12: "PhC-C2DL-PSC",
}
SCALED_DATASET = [
    "DIC-C2DH-HeLa",
    "Fluo-C2DL-MSC",
    "Fluo-N2DH-GOWT1",
    "Fluo-N2DH-SIM+",
    "Fluo-N2DL-HeLa",
    "PhC-C2DH-U373",
    "PhC-C2DL-PSC",
]

EXCLUDE_LIST = {"BF-C2DL-HSC": {1: [], 2: []},
           "BF-C2DL-MuSC": {1: [], 2: []},
           "DIC-C2DH-HeLa": {1: [67], 2: []},
           "Fluo-C2DL-MSC": {1: [9, 28, 30, 31, 36, 46, 47], 2: []},
           "Fluo-N2DH-GOWT1": {
               1: [2, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 27, 28, 29, 32, 34, 37, 40, 41, 44, 45, 46, 47],
               2: [22, 25, 27, 28, 29, 30, 35, 39, 46, 47, 60, 65, 76, 80, 82, 91]},
           "Fluo-N2DH-SIM+": {1: [], 2: []},
           "Fluo-N2DL-HeLa": {
               1: [12, 14, 15, 20, 21, 22, 23, 25, 29, 38, 39, 40, 44, 50, 51, 53, 54, 55, 62, 76, 77, 78, 79, 80, 81,
                   88], 2: [23, 35, 36, 67, 78, 87]},
           "PhC-C2DH-U373": {1: [], 2: []},
           "PhC-C2DL-PSC": {1: [], 2: []}, }


if __name__ == '__main__':
    train_list = {}
    for dataset in MODES.values():
        train_list[dataset] = {}
        for seq in [1, 2]:
            train_list[dataset][seq] = {}
            train_list[dataset][seq]["img_paths"] = []
            train_list[dataset][seq]["likeli_paths"] = []
            if dataset in SCALED_DATASET:
                scale_op = "-S"
            else:
                scale_op = ""
            mask_paths = sorted(Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/SEG").glob("*.tif"))
            likeli_paths = sorted(Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}_GT/LIK{scale_op}").glob("*.*"))
            img_paths = sorted(Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{dataset}/{seq:02d}{scale_op}").glob("*.*"))

            mask_list = []
            for mask_path in mask_paths:
                mask_list.append(int(mask_path.stem[-3:]))

            like_list = []
            for likeli_path in likeli_paths:
                like_list.append(int(likeli_path.stem))

            for idx in mask_list:
                try:
                    train_idx = like_list.index(idx)
                    train_list[dataset][seq]["img_paths"].append(img_paths[train_idx])
                    train_list[dataset][seq]["likeli_paths"].append(likeli_paths[train_idx])
                except:
                    train_list[dataset][seq]["img_paths"].append(img_paths[idx-50])
                    train_list[dataset][seq]["likeli_paths"].append(likeli_paths[idx-50])


    print(train_list)
    with Path("/home/kazuya/main/WSISPDR/detection/det_train.pickle").open("wb") as f:
        pickle.dump(train_list, f)


