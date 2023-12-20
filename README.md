# RLGC

 This program is stable under Python=3.7!

 We recommend that using conda before installing all the requirements. The details of our local conda environment are in:

 - environment.yaml

 If your local dependencies are the same as us, then you can run this command to setup your environment:

 - conda env create -f environment.yaml

 If not, you can first create a python3.7 environment and running this command:

 - pip install -r requirement.txt




 Directories and files included in the implementation:

 'models' - The well-trained RLGC models. 

 'samples' - 14 samples images from our testing set, which can be used to run a sample test of RLGC with its well-trained models.
 
 'iid_early_stop' - The folder to save the original and attacked inpainting images as well as their corresponding predicted masks.


 The commands for training and testing are:
 - Training:
 - CUDA_VISIBLE_DEVICES=0,1 python Train_torch.py
 - Testing:
 - CUDA_VISIBLE_DEVICES=0,1 python Tst.py