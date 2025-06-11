# Title 'Multi-scale KNN-based graph convolutional aggregation network with text prompt for jaw cyst segmentation

Integrating text prompt and graph attention convolutional network into multi-scale aggregation network for jaw cyst segmentation

The corrsponding dataset can be downloaded from the website: https://pan.baidu.com/s/13KMnDlZWgeyzq0XVI10j6Q and password: ztzu.

The pretrained model will be released at the the Baidu disk.

We use the nnUNetv2 framework, you can download for the website: https://github.com/MIC-DKFZ/nnUNet.

The above files except for nnUNetTrainer.py are put into the fold:nnunetv2\training\nnUNetTrainer\variants\network_architecture.

In the meanwhile, you replace the nnUNetTrainer class file with uploaded file named as 'nnUNetTrainer.py.



# How to use this code

1. Download nnUNet framework from the website:https://github.com/MIC-DKFZ/nnUNet
Notice: This code is based on the first version of nnUNet code. You must find the corresponding version, then you can use the command:

cd nnUNet
pip install -e .

3. Download the jaw cyst dataset from https://pan.baidu.com/s/13KMnDlZWgeyzq0XVI10j6Q and password: ztzu.

4. The above files are put into the fold: nnunetv2\training\nnUNetTrainer\variants\network_architecture.

5. The file 'nnUNetTrainer.py' is replaced, which is located in the folder 'nnunetv2\training\nnUNetTrainer\'.

6. put the file named as 'run_training.py' into the folder 'nnunetv2\run'

7. run the command 'python run_training.py'.


