# Synthetic image generative model
This repository is following the internship idea to generate synthetic data using cGANs. The model is implemented currently trained on TEM samples and their axon-myelin mask as input. The training dataset is preprocessed before being passed as input, the TEM samples are divided into 18 tile and each tile is merged with it's segementation mask side by side.The test dataset consists of SEM samples that have also been preprocessed similarly as training dataset to provide the sample and it's ground truth. The resulting output are TEM samples.
 </br>
On running ```cgan_tem_model.py``` , when specified (with the variable i) in the code. It will either train the model ```(i=1)``` or load a pre-exisiting checkpoint ```(i=1)```.
</br>
To test the data and it's segementation usign AxonDeepSeg with a zoom factor ```-z 2.0 ```.
</br>
</br>
The sequence to train the model/load a checkpoint is:
</br>
| Seq | Filename    | Function   |
| :---:   | :---: | :---: |
| 1 | tem_preprocessor.py  | Divides the dataset into train and val |
| 2 | split_rename.py  | Splits the input(TEM) into 18 tiles  |
| 3 | sem_dets.py  | Splits the input(SEM) |
| 4 | combine_A_and_B.py  | From the original pix_2_pix git repo  |
| 5 | cgan_tem_model.py  |  Train the generative model  |

</br>


