import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def blend_img(image, label, n):
    transform1 = torchvision.transforms.ToPILImage()
    transform2 = torchvision.transforms.PILToTensor()
    if n>1:
        blen_ten = torch.zeros((n,image.shape[1], image.shape[2], image.shape[3]))
        for i in range(0,n):
            image1 = image[i]
            label1 = label[i]
            label1 = 255 - label1
            img_pil = transform1(image1)
            lbl_pil = transform1(label1)
            blen_img = Image.blend(img_pil, lbl_pil, 0.4)
            blen_ten[i] = transform2(blen_img)
            # blen_ten[i] = blen_ten_ind
        blen_ten1 =blen_ten.to(dtype=torch.float32)


    else:
        image1 = image[0]
        label2 = label[0]
        label1 = 255-label2
        img_pil = transform1(image1)
        lbl_pil = transform1(label1)
        blen_img = Image.blend(img_pil, lbl_pil, 0.4)
        blen_ten = transform2(blen_img)
        blen_ten1 = blen_ten.unsqueeze(0).to(dtype=torch.float32) 

    # target = [n, image.shape[1], image.shape[2], image.shape[3]]
    # blen_ten1 = blen_ten[None, :, :, :].expand(target).to(dtype=torch.float32) 
    return blen_ten1

def conv_pil(image):
    transform1 = torchvision.transforms.ToPILImage()
    img_pil = transform1(image)
    return img_pil

def read_image_sem(image):
    label_img = image[:, :, :64]
    input_img = image[:, :, 64:]

    # input_img = np.array(input_img)
    # label_img = np.array(label_img)
    

    # label_img = label_img.astype(np.float32)
    # input_img = input_img.astype(np.float32)

    return input_img, label_img


class Div_Images(object):
    def __call__(self, image):
        image_a, image_b = read_image_sem(image)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)), #args.image_size = 64
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        lbl = image_b
        img = transform(image_a)
        res_img = torch.cat((lbl,img), 2)

        return res_img

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def get_data_lbl(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Div_Images()

    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def setup_logging_sample(run_name):
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
