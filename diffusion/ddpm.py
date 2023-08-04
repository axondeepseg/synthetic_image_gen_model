import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules_mod import UNet
from logger import Logger
import logging
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import save_image


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,  img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)


    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, noisy_im=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            if noisy_im==None:
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            else:
                x = noisy_im
            # print("noise steps",self.noise_steps)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        transform = transforms.Grayscale()
        x = transform(x)
        return x

def div_image(img):
    label_img = img[:, :, :64]
    input_img = img[:, :, 64:]
    return label_img,input_img


def train(args):
    logger = Logger(dir = "Log/", output_formats = "log")
    logger.log(f"Training diffusion model on dataset...")
    setup_logging(args.run_name)
    device = args.device
    logger.log(f"args: {args}")
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger1 = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    epoch_loss = []
    
    

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger1.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        #store loss values for plotting
        epoch_loss.append(loss.item())
        logger.log(f"For Epoch {epoch}, MSE: {loss.item()}")
        print("t = ",t)

            
        print(images.size())
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

    # print(epoch_loss)


    # Visualize loss history
    epoch_count = range(0, epoch+1)
    # print(epoch_count)
    plt.plot(epoch_count, epoch_loss, label='Training Loss')
    plt.title('Uncondtional DDPM TEM 900 epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('DDPM_Uncondtional_TEM_Epoch_900_lr_1e-4_normalised.png')


def translate(args):
    setup_logging_sample(args.run_name)
    device = args.device
    #load data
    dataloader = get_data(args)
    #load source model
    source_model = UNet().to(device)
    source_ckpt = torch.load("models/DDPM_Uncondtional_TEM_Epoch_900_lr_1e-4/ckpt.pt")
    source_model.load_state_dict(source_ckpt)
    #load target model
    target_model = UNet().to(device)
    target_ckpt = torch.load("models/DDPM_Uncondtional_SEM_Epoch_900_lr_1e-4/ckpt.pt")
    target_model.load_state_dict(target_ckpt)

    diffusion = Diffusion(img_size=64, device=device)

    logger1 = SummaryWriter(os.path.join("runs", args.run_name))
    
    logging.info(f"Start adding noise")
    pbar = tqdm(dataloader)
    for i, (images, _) in enumerate(pbar):
        images = images.to(device)
        x_img = (images * 255).type(torch.uint8)
        transform = transforms.Grayscale()
        x_img = transform(x_img)
        save_images(x_img, os.path.join("results", args.run_name, f"org{i}.jpg"))
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)
        predicted_noise = source_model(x_t, t)
        x = (predicted_noise * 255).type(torch.uint8)
        transform = transforms.Grayscale()
        x = transform(x)
        save_images(x, os.path.join("results", args.run_name, f"noise{i}.jpg"))
        x_trans = diffusion.sample(target_model, n = images.shape[0], noisy_im = predicted_noise )
        save_images(x_trans, os.path.join("results", args.run_name, f"trans{i}.jpg"))




    


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional_TEM_Epoch_900_lr_1e-4"
    args.epochs = 900
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"data_tem_diff_256/B/train"
    args.device = "cuda"
    args.lr = 1e-4
    train(args)
    
def translate_im():
    #TRANSLATE IMAGE
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_TEM_to_SEM"
    args.epochs = 900
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"diff_data/A"
    args.device = "cuda"
    args.lr = 1e-4
    translate(args)
    

if __name__ == '__main__':
    # launch()
    #Loads TEM model and generates synthetic images
    device = "cuda"
    logger.log(f"Sampling Model...")
    run_name = "DDPM_Uncondtional_TEM_Epoch_2_lr_1e-4_test"
    setup_logging_sample(run_name)
    model = UNet().to(device)
    ckpt = torch.load("models/DDPM_Uncondtional_TEM_Epoch_900_lr_1e-4/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    c = 0
    for i in range(20):
        x = diffusion.sample(model, 1)
        c = i + 10
        save_images(x, os.path.join("results", run_name, f"{c}.jpg"))

    
