"""
This script splits each image in the sem directory
into hopefully 2 tiles each tiles and renames the files
"""

from AxonDeepSeg import ads_utils
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os
import shutil
import imageio
import tensorflow as tf
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def split18(image):
    '''Splits an image in 18 tiles.'''
    heigth, width = image.shape
    transform = transforms.ToTensor()
    # Convert the image to PyTorch tensor
    te = transform(image)
    if(te.shape[1] > 762):
        row = te.shape[1]
        diff = row - 762
        te1 = te[:, diff:, :]
        te = te1
    elif(te.shape[1] < 762):
        row = te.shape[1]
        diff = 762 - row
        pad = (0, 0, diff, 0)
        te1 = F.pad(te, pad, "constant", 0)
        te = te1
    else:
        print("same size")
    
    
    if(te.shape[2] > 1254):
        row = te.shape[2]
        diff = row - 1254
        te1 = te[:, :, diff:]
        te = te1
    elif(te.shape[2] < 1254):
        row = te.shape[2]
        diff = 1254 - row
        pad = (diff, 0, 0, 0)
        te1 = F.pad(te, pad, "constant", 0)
        te = te1
    else:
        print("same size")
    return te

def split2(te):
    im1 = te[:, :, :627]
    im2 = te[:, :, 627:]
    # print(tiles.shape)
    te12 = np.array(im1)
    te13 = te12[0, :, :]
    img1 = imageio.core.util.Array(te13)
    print(img1.shape)
    te22 = np.array(im2)
    te23 = te22[0, :, :]
    img2 = imageio.core.util.Array(te23)
    print(img2.shape)
    return img1, img2

def split_rename_images(directory):
    ''' Splits and rename every image in directory.'''
    operating_dir = os.getcwd()
    os.chdir(directory)
    imgs = sorted(os.listdir('.'))
    oth_file = 1
    filename = 1
    pbar = tqdm(total=len(imgs))
    for img_name in imgs:
        img = ads_utils.imread(img_name)
        img_pd = split18(img)
        im1, im2 = split2(img_pd)
        img_pdd = imageio.core.util.Array(img_pd)
        ads_utils.imwrite(f'{str(oth_file)}.png', im1)
        oth_file += 1
        # for i in range(2):
        #     for j in range(19):
        # tile = tiled_array[i,j,:,:]
        ads_utils.imwrite(f'{str(filename)}.png', im1)
        filename += 1
        ads_utils.imwrite(f'{str(filename)}.png', im2)
        filename += 1
        os.remove(img_name)
        pbar.update(1)
    pbar.close()
    os.chdir(operating_dir)

def main():
    path_at, path_av = 'data_sem3/A2/train', 'data_sem3/A2/val'
    path_bt, path_bv = 'data_sem3/B2/train', 'data_sem3/B2/val'
    for d in [path_av,path_bt,path_bv]:
        split_rename_images(d)
        print(f'Done with {d}.')


if __name__ == '__main__':
    main()


	
