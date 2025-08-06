# Title 'Multi-scale KNN-based graph convolutional aggregation network with text prompt for jaw cyst segmentation

Integrating text prompt and graph attention convolutional network into multi-scale aggregation network for jaw cyst segmentation

The corrsponding dataset can be downloaded from the website: https://pan.baidu.com/s/13KMnDlZWgeyzq0XVI10j6Q.

# Note: You can download the datasets from the URLS before you send the Dataset End-User Agreement Instructions, we will send the password or the updating urls to you.

The pretrained model will be released at the the Baidu disk.

We use the nnUNetv1 framework, you can download the old code nnUNetv1 for the website: https://github.com/MIC-DKFZ. If you cann't find the old version of nnUNet, you also contact me.

The above files except for nnUNetTrainer.py are put into the fold:nnunetv2\training\nnUNetTrainer\variants\network_architecture.

In the meanwhile, you replace the nnUNetTrainer class file with uploaded file named as 'nnUNetTrainer.py.



# How to use this code

1. Download nnUNet framework from the website:https://github.com/MIC-DKFZ/nnUNetv1
Notice: This code is based on the first version of nnUNet code. You must find the corresponding version, then you can use the command:

cd nnUNet
pip install -e .

3. Download the jaw cyst dataset from https://pan.baidu.com/s/13KMnDlZWgeyzq0XVI10j6Q(We will update it anytime, and the final email shall prevail).

4. The above files are put into the fold: nnunet\training\nnUNetTrainer\variants\network_architecture.

5. The file 'nnUNetTrainer.py' is replaced, which is located in the folder 'nnunet\training\nnUNetTrainer\'.

6. put the file named as 'run_training.py' into the folder 'nnunetv2\run'

7. run the command 'python run_training.py'.


