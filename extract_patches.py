import re
import glob
import tqdm
import pathlib
import os
import numpy as np
import datetime
from patch_extractor import PatchExtractor
from utils import rm_n_mkdir
from dataset import get_dataset

if __name__ == "__main__":
    current_time = datetime.now().strftime('%Y_%M_%D--%H_%M_%S')
    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True



    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "cpm15"
    save_root = f"datasets/training_data/{dataset_name}/current_time"

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".png", "G:/hovernet/datasets/cpm15/Images"),
            "ann": (".mat", "G:/hovernet/datasets/cpm15/Labels"),
        },
        "valid": {
            "img": (".png", "G:/hovernet/datasets/cpm15/Images"),
            "ann": (".mat", "G:/hovernet/datasets/cpm15/Labels"),
        },
    }
    parser = get_dataset(dataset_name)

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.
    xtractor = PatchExtractor(win_size, step_size, True)

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)  # 一个函数 '[123]' => '[[]123[]]



    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]  # '.png'  '文件夹地址'
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = f"{save_root}/{split_name}/{win_size[0]}x{win_size[1]}_{step_size[0]}x{step_size[1]}"
        # 上面的都ok

        # print('ann_dir:', ann_dir)
        # print('ann_ext:', ann_ext)
        file_list = glob.glob(patterning(f"{ann_dir}/*{ann_ext}"))
        file_list.sort()  # ensure same ordering across platform
        # print(file_list)
        # print('lists: ', os.listdir(ann_dir))
        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"  # 普通string

        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img(f"{img_dir}/{base_name}{img_ext}")
            # ann = parser.load_ann(f"{ann_dir}/{base_name}{ann_ext}", type_classification)
            ann = parser.load_ann(f"{ann_dir}/{base_name}{ann_ext}")

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)  # extract_type = mirror

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()