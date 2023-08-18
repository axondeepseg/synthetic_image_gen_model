"""
This script splits each image in the data_tem directory
into 18 tiles and renames the files
"""

from AxonDeepSeg import ads_utils
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os
import shutil


def split18(image):
    '''Splits an image in 18 tiles.'''
    heigth, width = image.shape
    tiles = image.reshape(3, heigth // 3, 6, width // 6).swapaxes(1,2)
    print(tiles.shape)
    return tiles

def split_rename_images(directory):
    ''' Splits and rename every image in directory.'''
    operating_dir = os.getcwd()
    os.chdir(directory)
    imgs = sorted(os.listdir('.'))
    filename = 1
    pbar = tqdm(total=len(imgs))
    for img_name in imgs:
        img = ads_utils.imread(img_name)
        tiled_array = split18(img)
        for i in range(11):
            for j in range(19):
                tile = tiled_array[i,j,:,:]
                ads_utils.imwrite(f'{str(filename)}.png', tile)
                filename += 1
        os.remove(img_name)
        pbar.update(1)
    pbar.close()
    os.chdir(operating_dir)

def main():
    path_at, path_av = 'data_tem/A/train', 'data_tem/A/val'
    path_bt, path_bv = 'data_tem/B/train', 'data_tem/B/val'
    for d in [path_at, path_av, path_bt, path_bv]:
        split_rename_images(d)
        print(f'Done with {d}.')


if __name__ == '__main__':
    main()
