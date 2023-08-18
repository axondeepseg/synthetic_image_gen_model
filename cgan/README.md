# Conditional Generative Adversarial Network (cGAN)
## Training cGAN for Image-to-Image Translation
A cGAN model can be trained to translate an image of a certain contrast to another by running the Python file `cgan_tem_model.py`. The file is tailored to train the model or run a model after training with its weights. 
</br>
This is controlled by the variable `i` in the file if it is set to 0 or 1. On default, the model is set to 0 to load a pre-existing checkpoint rather than train the model.
</br>
To use the pre-existing checkpoints, download the assets present in the `v1.0` release. 
</br>
To train to translate TEM contrast to SEM contrast run,
```
 python cgan_tem_model.py
```

To train to translate SEM contrast to TEM contrast run,
```
 python cgan_sem_model.py
 ```
To test the data and its segmentation using AxonDeepSeg with a zoom factor -z 2.0.
