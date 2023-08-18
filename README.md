# Synthetic image generative model
## Introduction
High-resolution pathology images of axons and myelin are used to observe demyelination on neurons. These images are captured through an electron microscope. Different types of microscopy reflect different properties on the neurons and are known as contrasts. The two main contrasts are known as Transmission EM (TEM) and Scanning EM (SEM). This project aims to perform unpaired image-to-image translation from one contrast to another to increase our overall training dataset. We implement various models such as CGANs and DDPM to achieve translation.
</br>
</br>
## Data
The data used in this project is the ```data_axondeepseg_tem``` and ```data_axondeepseg_sem```datasets privately hosted on the Neuropoly server with git-annex. ```data_axondeepseg_tem``` contains 20 subjects of TEM images used for axon-myelin segmentation. ```data_axondeepseg_sem``` contains 8 subjects of SEM images used for axon-myelin segmentation.
</br>
These images are further processed into 64x64 tiles for training. Instructions on processing is described in detail in the `preprocessing` folder. 
</br>
</br>
## Models
### Conditional Generative Adversarial Network (CGAN)
Generative Adversarial Network is a DL network used to generate synthetic data. It consists of two models - Generator (G) and Discriminator (D). The aim is for the G to pass synthetic images to D and continue training until it is able to generate a realistic image and pass it through D. 
Conditional GAN is a type of GAN that uses labels to aid in training are generating data closer to the domain. In this project, the labels are segmentation axon-myelin masks produced by AxonDeepSeg. 
</br>
While these models are used to generate synthetic data they can also aid in image-to-image transition. The implemented model is known as pix2pix where the generator used is a U-Net model and the discriminator is a convolutional PatchGAN. 
</br>
</br>
This model is implemented with further instruction in the README in the ```cgan``` folder with model checkpoints available in ```v1.0```assets.
</br>
</br>
### Denoising Diffusion Probabilistic Models (DDPM)
DDPM is a diffusion-based model which executes the idea of discrete denoising to generate synthetic data. Data is represented in a Markov Chain over the span of T timesteps where the model gradually noises/denoises an image to learn the probability distribution of the domain. The training consists of two processes:
</br>
1. Forward Process: The training image has Gaussian noise added over a series of timesteps until it contains 100% Gaussian noise. These noisy images are targets for the neural network. </br>
2. Reverse or Sampling Process: Neural network is trained to  recover the original data by de-noising the images. Training occurs in this process, learning the probability distribution of the domain.
</br>
</br>
Reproduction of this model is described in the README in the `diffusion` folder with model checkpoints available in `v2.0` assets.
