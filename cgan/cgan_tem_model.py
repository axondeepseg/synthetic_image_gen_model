import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import functools
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from skimage import io, transform
from torch.autograd import Variable
from torchvision.utils import save_image
from torchsummary import summary
from pathlib import Path

import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 4,5,6,7'

IMG_WIDTH = 256
IMG_HEIGHT = 256

'''Data Preprocessing'''

# splits the GT and org image, hardcoded the padding to 768x768 so that on maxpool it reduces to 256x256.
def read_image(image):
        
    #image = np.array(image)
    #image = tf.convert_to_tensor(image)
    #width = image.shape[1]
    #print(image.shape)
    
    input_image = image[:, :, :627]
    target_image = image[:, :, 627:]

    pad1 = (70,71,3,3) # pad last dim by 1 on each side
    # np.pad(target_image, [(70, 71), (3, 3)], 'constant', constant_values=0)
    # np.pad(input_image, [(70, 71), (3, 3)], 'constant', constant_values=0)


    target_image = F.pad(target_image, pad1, "constant", 0)
    input_image = F.pad(input_image, pad1, "constant", 0)

    p = nn.MaxPool2d(3, stride=3)
    y = p(input_image)
    x = p(target_image)

    target_image = x
    input_image = y

    #print(target_image.shape)
    #print(input_image.shape)
    target_image = np.array(target_image)
    input_image = np.array(input_image)
    

    input_image = input_image.astype(np.float32)
    target_image = target_image.astype(np.float32)

    return input_image, target_image

def read_image_sem(image):
        
    #image = np.array(image)
    #image = tf.convert_to_tensor(image)
    #width = image.shape[1]
    #print(image.shape)
    
    input_image = image[:, :, :627]
    target_image = image[:, :, 627:]

    #print(target_image.shape)
    #print(input_image.shape)
    target_image = np.array(target_image)
    input_image = np.array(input_image)
    

    input_image = input_image.astype(np.float32)
    target_image = target_image.astype(np.float32)

    return input_image, target_image

        
def normalize(inp, tar):
    input_image = (inp / 127.5) - 1
    target_image = (tar / 127.5) - 1
    return input_image, target_image
        
class Train_Normalize(object):
    def __call__(self, image):
        #print("one sample")
        inp, tar = read_image(image)
        #print(inp.shape)
        #inp, tar = random_jittering_mirroring(inp, tar)
        #print(inp.shape)
        inp, tar = normalize(inp, tar)
        #print(inp.shape)
        #image_a = torch.from_numpy(inp.copy().transpose((2,0,1)))

        image_a = torch.from_numpy(inp.copy())
        image_b = torch.from_numpy(tar.copy())
        #print(image_a.shape)
        return image_a, image_b  

class Sem_Normalize(object):
    def __call__(self, image):
        #print("one sample")
        inp, tar = read_image_sem(image)
        #print(inp.shape)
        #inp, tar = random_jittering_mirroring(inp, tar)
        #print(inp.shape)
        inp, tar = normalize(inp, tar)
        #print(inp.shape)
        #image_a = torch.from_numpy(inp.copy().transpose((2,0,1)))

        image_a = torch.from_numpy(inp.copy())
        image_b = torch.from_numpy(tar.copy())
        #print(image_a.shape)
        return image_a, image_b    
    
class Val_Normalize(object):
    def __call__(self, image):
        inp, tar = read_image(image)
        #inp, tar = random_jittering_mirroring(inp, tar)
        inp, tar = normalize(inp, tar)
        image_a = torch.from_numpy(inp.copy())
        image_b = torch.from_numpy(tar.copy())
        return image_a, image_b

DIR = 'data_tem/AB/train/'
n_gpus = 1
#batch_size = 32 * n_gpus

train_ds = ImageFolder(DIR, transform=transforms.Compose([
        transforms.Resize([762,1254]),
        transforms.ToTensor(),
        Train_Normalize()]))
train_dl = DataLoader(train_ds)

VAL_DIR = 'data_tem/AB/val/'

#batch_size = 32 * n_gpus

val_ds = ImageFolder(VAL_DIR, transform=transforms.Compose([
        transforms.Resize([762,1254]),
        transforms.ToTensor(),
        Val_Normalize()]))
val_dl = DataLoader(val_ds)

TEST_DIR = 'data_tem/AB/test/'
test_ds = ImageFolder(TEST_DIR, transform=transforms.Compose([
        transforms.Resize([762,1254]),
        transforms.ToTensor(),
        Val_Normalize()]))
test_dl = DataLoader(test_ds)

SEM_DIR = 'data_sem3/AB3/'
sem_ds = ImageFolder(SEM_DIR, transform=transforms.Compose([
        transforms.Resize([762,1254]),
        transforms.ToTensor(),
        Val_Normalize()]))
sem_dl = DataLoader(sem_ds)


def imshow(inputs, target, figsize=(10, 5)):
    # print(inputs.shape)
    # print(target.shape)

    inputs = np.uint8(inputs)
    target = np.uint8(target)
    tar = np.rollaxis(target[0], 0, 3)
    inp = np.rollaxis(inputs[0], 0, 3)
    title = ['Input Image', 'Ground Truth']
    display_list = [inp, tar]
    plt.figure(figsize=figsize)
  
    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.axis('off')
        plt.imshow(display_list[i])
    plt.axis('off')
 
    #plt.imshow(image)    

#to display the dataset
def show_batch(dl):
    j=0
    for (images_a, images_b), _ in dl:
        j += 1
        imshow(images_a, images_b)
        if j == 3:
            break

#show_batch(train_dl)

# custom weights initialization called on generator and discriminator        
def weights_init(net, init_type='normal', scaling=0.02):
    def init_func(m):  
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif classname.find('BatchNorm2d') != -1: 
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    #print('initialize network with %s' % init_type)
    net.apply(init_func)  


def get_norm_layer():
    norm_type = 'batch'
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer

class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, nf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
    
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        
        # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        # gradually reduce the number of filters from nf * 8 to nf
        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):


    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
    
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.device_count()
torch.cuda.set_device('cuda:0')
norm_layer = get_norm_layer()

generator = UnetGenerator(3, 3, 64, norm_layer=norm_layer, use_dropout=False)#.cuda().float()
generator.apply(weights_init)

generator = torch.nn.DataParallel(generator) 

inp = torch.ones(1, 3,256,256)
#gen = generator(inp)
device = 'cuda'
inp = inp.to(device)
generator = generator.to(device)

#print(summary(generator,(3,256,256)))

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

discriminator = Discriminator(6, 64, n_layers=3, norm_layer=norm_layer)#.cuda().float()
discriminator.apply(weights_init)
discriminator = torch.nn.DataParallel(discriminator)  # multi-GPUs

inp = torch.ones(1,6,256,256)
discriminator = discriminator.to(device)
disc = discriminator(inp)

#summary(discriminator,(6,256,256))

adversarial_loss = nn.BCELoss() 
l1_loss = nn.L1Loss()

def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    #print(gen_loss)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

def save_ckp1(state, is_best, checkpoint_dir, best_model_dir):
    f_path = os.path.join(checkpoint_dir, "checkpoint_gen.pt")
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, "best_model_gen.pt")
        shutil.copyfile(f_path, best_fpath)

def save_ckp2(state, is_best, checkpoint_dir, best_model_dir):
    f_path = os.path.join(checkpoint_dir, "checkpoint_dis.pt")
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, "best_model_dis.pt")
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

learning_rate = 2e-4 
G_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate, betas=(0.5, 0.999))

c_check = 1
num_epochs = 150
D_loss_plot, G_loss_plot = [], []
i= 0
if i==1:
    print("Training Model")
    for epoch in range(1, num_epochs+1): 
    

        D_loss_list, G_loss_list = [], []
    
        for (input_img, target_img), _ in train_dl:
        
            D_optimizer.zero_grad()
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            
    #         print("Inp:",input_img.shape)
    #         print("Tar:", target_img.shape)
            
            generated_image = generator(input_img)
            
            disc_inp_fake = torch.cat((input_img, generated_image), 1)
        
            
            
            real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
            fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
            
    #         print("Fake_targ:",fake_target.shape)
            D_fake = discriminator(disc_inp_fake.detach())
            
    #         print("D_fake:",D_fake.shape)
            D_fake_loss   =  discriminator_loss(D_fake, fake_target)
            # print(discriminator(real_images))
            #D_real_loss.backward()
        
            disc_inp_real = torch.cat((input_img, target_img), 1)
            
                                            
            output = discriminator(disc_inp_real)
            D_real_loss = discriminator_loss(output,  real_target)

        
        
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_loss_list.append(D_total_loss)
        
            D_total_loss.backward()
            D_optimizer.step()
        
            G_optimizer.zero_grad()
            fake_gen = torch.cat((input_img, generated_image), 1)
    #         print('fake_gen:', fake_gen)
            G = discriminator(fake_gen)
            G_loss = generator_loss(generated_image, target_img, G, real_target)                                 
            G_loss_list.append(G_loss)

            G_loss.backward()
            G_optimizer.step()
            
        print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (
                (epoch), num_epochs, torch.mean(torch.FloatTensor(D_loss_list)),\
                torch.mean(torch.FloatTensor(G_loss_list))))
        
        D_loss_plot.append(torch.mean(torch.FloatTensor(D_loss_list)))
        G_loss_plot.append(torch.mean(torch.FloatTensor(G_loss_list)))

        checkpoint_dir = "tem_gan/new_checkpnt_150ep/" 
        model_dir = "tem_gan/new_checkpnt_150ep"
        is_best = False
        if epoch== 1:
            train_input = target_img
            train_inp = train_input.view(train_input.shape[1], train_input.shape[2], train_input.shape[3])
                # fake2 = fake11.view(fake11.shape[1], fake11.shape[2], fake11.shape[3])
            save_image(train_inp, 'tem_gan/img_new_150ep/train_inp.png', normalize=True)
            checkpoint_g = {
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'optimizer': G_optimizer.state_dict()
            }
            save_ckp1(checkpoint_g, is_best, checkpoint_dir, model_dir)
            checkpoint_d = {
                'epoch': epoch + 1,
                'state_dict': discriminator.state_dict(),
                'optimizer': D_optimizer.state_dict()
            }
            save_ckp2(checkpoint_d, is_best, checkpoint_dir, model_dir)
        if epoch== 100:
            checkpoint_g = {
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'optimizer': G_optimizer.state_dict()
            }
            save_ckp1(checkpoint_g, is_best, checkpoint_dir, model_dir)
            checkpoint_d = {
                'epoch': epoch + 1,
                'state_dict': discriminator.state_dict(),
                'optimizer': D_optimizer.state_dict()
            }
            save_ckp2(checkpoint_d, is_best, checkpoint_dir, model_dir)
        if epoch== 150:
            is_best = True
            checkpoint_g = {
                'epoch': epoch + 1,
                'state_dict': generator.state_dict(),
                'optimizer': G_optimizer.state_dict()
            }
            save_ckp1(checkpoint_g, is_best, checkpoint_dir, model_dir)
            checkpoint_d = {
                'epoch': epoch + 1,
                'state_dict': discriminator.state_dict(),
                'optimizer': D_optimizer.state_dict()
            }
            save_ckp2(checkpoint_d, is_best, checkpoint_dir, model_dir)
        # torch.save(generator.state_dict(), 'tem_gan/generator_epoch_%d.pth' % (epoch))
        # torch.save(discriminator.state_dict(), 'tem_gan/discriminator_epoch_%d.pth' % (epoch))
        
        for (inputs, targets), _ in val_dl:  
            inputs = inputs.to(device)
            generated_output = generator(inputs)
            save_image(generated_output.data[:10], 'tem_gan/img_new_150ep/val_sample_%d'%epoch + '.png', nrow=5, normalize=True)

        for (real, condition), _ in test_dl:
                fake = generator(real)
                cond = real
                rel = condition
                # p1 = nn.MaxUnpool2d(3, stride=3)
                # fake = p1(fake11)
                cond1 = cond.view(cond.shape[1], cond.shape[2], cond.shape[3])
                real1 = rel.view(rel.shape[1], rel.shape[2], rel.shape[3])
                fake1 = fake.view(fake.shape[1], fake.shape[2], fake.shape[3])
                # fake2 = fake11.view(fake11.shape[1], fake11.shape[2], fake11.shape[3])
                save_image(fake1, 'tem_gan/img_new_150ep/SampleFake_%d'%epoch  + '.png', normalize=True)
                # save_image(fake2, 'tem_gan2/samples1500/SampleFake_%d'%epoch  + '.png', normalize=True)

                if c_check==1:
                    save_image(cond1, 'tem_gan/img_new_150ep/SampleCon_%d'%epoch  + '.png', normalize=True)
                    save_image(real1, 'tem_gan/img_new_150ep/SampleReal_%d'%epoch  + '.png', normalize=True)
                c_check = c_check+1 

if i==0:
    print("Loading Checkpoint")
    model_G = generator
    model_D = discriminator
    optim_G = G_optimizer 
    optim_D = D_optimizer 
    ckp_path_dis = "tem_gan/new_checkpnt_150ep/checkpoint_dis.pt"
    ckp_path_gen = "tem_gan/new_checkpnt_150ep/checkpoint_gen.pt"
    model_G, optim_G, start_epoch1= load_ckp(ckp_path_gen, model_G, optim_G) 
    model_D, optim_D, start_epoch2 = load_ckp(ckp_path_dis, model_D, optim_D) 
    # print("model = ", model_G)
    # print("optimizer = ", optim_G)
    # print("start_epoch = ", start_epoch1)
    # print("valid_loss_min = ", valid_loss_min_D)
    # print("valid_loss_min = {:.6f}".format(valid_loss_min))
    c_check = 1
    count = 1
    epoch = start_epoch1
    for (real, condition), _ in sem_dl:
        fake = generator(real)
        cond = real
        rel = condition
        # p1 = nn.MaxUnpool2d(3, stride=3)
        # fake = p1(fake11)
        cond1 = cond.view(cond.shape[1], cond.shape[2], cond.shape[3])
        real1 = rel.view(rel.shape[1], rel.shape[2], rel.shape[3])
        fake1 = fake.view(fake.shape[1], fake.shape[2], fake.shape[3])
        # fake2 = fake11.view(fake11.shape[1], fake11.shape[2], fake11.shape[3])
        save_image(fake1, 'tem_gan/test_150ep/test9/%d'%count  + '.png', normalize=True)
        save_image(cond1, 'tem_gan/test_150ep/test9/SampleCon_%d'%count  + '.png', normalize=True)
        save_image(real1, 'tem_gan/test_150ep/test9/SampleReal_%d'%count  + '.png', normalize=True)
        # save_image(fake2, 'tem_gan2/samples1500/SampleFake_%d'%epoch  + '.png', normalize=True)
        count  = count + 1
        if c_check==1:
            save_image(cond1, 'tem_gan/test_150ep/SampleCon_%d'%epoch  + '.png', normalize=True)
            save_image(real1, 'tem_gan/test_150ep/SampleReal_%d'%epoch  + '.png', normalize=True)
        c_check = c_check+1

    print("Inference Done")
    
