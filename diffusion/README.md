# Implementation of DDPM
DDPM models currently are meant to train on one dataset to create new synthetic images and is further used in image-to-image translation. The following section describes the process to run the DDPM model with existing checkpoints available in the `v2.0` Release. 
## Generate new synthetic images
1. Create ```models``` folder in directory.
2. Download Checkpoints of models from Release ```v2.0```.
3. Extract files of ```DDPM_Checkpoints``` into models.
4. Run the line
```
python ddpm.py
```

Results are stored in ```results```  in this directory.
