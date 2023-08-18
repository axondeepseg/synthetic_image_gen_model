# Preprocessing of the dataset
The various dataset involved in image-to-image translation consists of images of varying sizes. Images of these sizes are too large and take a long time to train a model. Hence, these images are divided into 64 x 64 patches. These images are also combined with their ground truth for easy reference to the ground truth during training.
</br>
To preprocess images, the following steps are defined below:
</br>
1. Activate AxonDeepSeg virtual environment 'ads_venv'
   ```
   conda activate ads_venv
   ```
2. Extract data and their ground truth and places them in folders A (Ground Truth) and B (Orginal Subject) along with dividing it into training and validation.
   ```
   python tem_preprocessor.py
   ```
3. Split the images into 64 x 64 tiles and rename them for easier processing.
   - Splitting TEM images
   ```
   python split_rename.py
   ```
   - Splitting SEM images
   ```
   python sem_split_rename.py
   ```
5. Combine GT with the original image into images of size 64 x 128 pixels into a new folder created called "AB".
   ```
   python combine_A_and_B.py
   ```
   
