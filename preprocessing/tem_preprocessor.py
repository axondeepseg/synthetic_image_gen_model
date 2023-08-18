"""
This script loads the TEM dataset and organizes the files so they 
can be fed to the pix2pix `combine_A_and_B.py` preprocessing script
The hierarchy should look like this:

data/
├── A/
│   ├── train
│   ├── val
├── B/
│   ├── train
│   └── val

"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import os
import shutil

#DATAPATH = "/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_tem"
DATAPATH = "data_axondeepseg_tem"


def split_list(l):
    return l[::2], l[1::2]

def main():
    # locating data and listing samples
    dataset_path = Path(DATAPATH)
    samples_path = dataset_path / 'samples.tsv'
    samples = pd.read_csv(samples_path, delimiter='\t')
    subject_dict = {}
    for i, row in samples.iterrows():
        subject = row['participant_id']
        sample = row['sample_id']
        if subject not in subject_dict:
            subject_dict[subject] = {}
        subject_dict[subject][sample] = {}

    # loading data
    sample_count = 0
    for subject in subject_dict.keys():
        samples = subject_dict[subject].keys()
        images_path = dataset_path / subject / "micr"
        images = list(images_path.glob('*.png'))
        masks_path = dataset_path / "derivatives" / "labels" / subject / "micr"
        masks = list(masks_path.glob('*axonmyelin*'))
        print(f"Looking at {len(samples)} samples...")
        for sample in samples:
            for img in images:
                if sample in str(img):
                    subject_dict[subject][sample]['image'] = str(img)
                    sample_count += 1
            for mask in masks:
                if sample in str(mask):
                    subject_dict[subject][sample]['mask'] = str(mask)
    # print(json.dumps(subject_dict, indent=4))
    print(f'{sample_count} samples collected.')

    # creating file tree (A for masks, B for images)
    path_at, path_av = 'data_tem/A/train', 'data_tem/A/val'
    path_bt, path_bv = 'data_tem/B/train', 'data_tem/B/val'
    [os.makedirs(path) for path in [path_at, path_av, path_bt, path_bv]]
    for subj in subject_dict:
        samples = list(subject_dict[subj].keys())
        # every subject has 50% of its samples in training set, 50% in validation set
        t_samples, v_samples = split_list(samples)
        for s in t_samples:
            mask = subject_dict[subj][s]['mask']
            shutil.copy(mask, path_at)
            img = subject_dict[subj][s]['image']
            shutil.copy(img, path_bt)
        for s in v_samples:
            mask = subject_dict[subj][s]['mask']
            shutil.copy(mask, path_av)
            img = subject_dict[subj][s]['image']
            shutil.copy(img, path_bv)


if __name__ == '__main__':
    main()
