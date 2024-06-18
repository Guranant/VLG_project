# CollegiateDataClub: Optimizing Data Processing and Analysis



The official PyTorch implementation of the paper
> [CollegiateDataClub: Optimizing Data Processing and Analysis](https://arxiv.org/abs/2401.15235) \
> Amirhosein Ghasemabadi, Muhammad Kamran Janjua, Mohammad Salameh, Chunhua Zhou, Fengyu Sun, Di Niu\
> Accepted at Transactions on Machine Learning Research (TMLR), 2024.

## Installation
This implementation is based on [BasicSR](https://github.com/xinntao/BasicSR) which is an open-source toolbox for image/video restoration tasks, [NAFNet](https://github.com/megvii-research/NAFNet), [Restormer](https://github.com/swz30/Restormer/tree/main/Denoising) and [Multi Output Deblur](https://github.com/Liu-SD/multi-output-deblur)

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## Quick Start 
We have provided ```demo-denoising.ipynb``` to show how to load images from the validation dataset and use the model to restore images.



## CollegiateDataClub implementation
The implementation of our proposed CollegiateDataClub Net, CollegiateDataClub block, and the Global Context Extractor module can be found in ```/CollegiateDataClub/basicsr/models/archs/CGNet_arch.py```

The implementation of Multi-Head CollegiateDataClub Net can be found in ```/CollegiateDataClub/basicsr/models/archs/CGNetMultiHead_arch.py```



<!-- ## Reproduce the Results -->

##  Denoising on SIDD
### 1. Data Preparation
##### Download the train set (from the SIDD dataset website) and place it in ```./datasets/SIDD/Data/```,
##### Download the evaluation data in lmdb format (from the Gopro dataset website) and place it in ```./datasets/SIDD/test/```:
#### After downloading, it should be like this:

```bash
./datasets/
└── SIDD/
    ├── Data/
    │   ├── 0001
    │   │   ├── GT_SRGB.PNG
    │   │   ├── NOISY_SRGB.PNG
    │   │   ....
    │   └── 0200
    │       ├── GT_SRGB.PNG
    │       ├── NOISY_SRGB.PNG    
    ├── train/
    └── test/
        ├── input.imdb
        └── target.imdb

```
* Use ```python scripts/data_preparation/sidd.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format. The processed images will be saved in ```./datasets/SIDD/train/```


### 2. Training

* To train the CollegiateDataClub model:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8081 basicsr/train.py -opt options/train/SIDD/CollegiateDataClub-SIDD.yml --launcher pytorch
```


### 3. Evaluation


#### Note: Due to the file size limitation, we are not able to share the pre-trained models in this code submission. However, they will be provided with an open-source release of the code.


##### Testing the model

  * To evaluate the pre-trained model use this command:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8080 basicsr/test.py -opt ./options/test/SIDD/CollegiateDataClub-SIDD.yml --launcher pytorch
```

### 4. Model complexity and inference speed
* To get the parameter count, MAC, and inference speed use this command:
```
python CollegiateDataClub/basicsr/models/archs/CGNet_arch.py
```

##  Gaussian Image denoising
### 1. Data Preparation
##### Clone the [Restormer's github project](https://github.com/swz30/Restormer/tree/main/Denoising) and follow their instructions to download the train and test datasets.

### 2. Training

#### To train the CollegiateDataClub model follow these steps:


* Copy the /CollegiateDataClub/basicsr/models/archs/CGNet_Guassian_arch.py to /Restormer/basicsr/models/archs/

* Copy the training option files from /CollegiateDataClub/options/train/Gaussian/ to /Restormer/Denoising/Options/

* Follow [Restormer's](https://github.com/swz30/Restormer/tree/main/Denoising) training instructions and train models on different noise levels



### 3. Evaluation


#### Note: Pretrained models will be released soon.


##### Testing the model

  * To evaluate the pre-trained model, start by adjusting the noise level (sigma=15, 25 or 50), the paths to the trained model, and the training option file within the code. Once modified, execute the following command.
```
python CollegiateDataClub/basicsr/test_gaussian_color_denoising.py
```

### 4. Model complexity and inference speed
* To get the parameter count, MAC, and inference speed use this command:
```
python CollegiateDataClub/basicsr/models/archs/CGNet_Guassian_arch.py
```


##  Deblurring on GoPro
### 1. Data Preparation
##### Download the train set (from the Gopro dataset website) and place it in ```./datasets/GoPro/train```,
##### Download the evaluation data in lmdb format (from the Gopro dataset website) and place it in ```./datasets/GoPro/test/```:
#### After downloading, it should be like this:

```bash
./datasets/
└── GoPro/
    ├── train/
    │   ├── input/
    │   └── target/
    └── test/
        ├── input.imdb
        └── target.imdb

```
* Use ```python scripts/data_preparation/gopro.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format.


### 2. Training

* To train the CollegiateDataClub Multihead model:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8081 basicsr/train.py -opt options/train/GoPro/CollegiateDataClubMH-GoPro.yml --launcher pytorch
```
* To finetune the trained CollegiateDataClubMH model on larger patches
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8081 basicsr/train.py -opt options/train/GoPro/CollegiateDataClubMH-GoPro-fintune_largerPatch.yml --launcher pytorch
```

### 3. Evaluation


#### Note: Pretrained models will be released soon.


##### Testing the model

  * To evaluate the pre-trained model use this command:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8080 basicsr/test.py -opt ./options/test/GoPro/CollegiateDataClubMH-GoPro.yml --launcher pytorch
```

### 4. Model complexity and inference speed
* To get the parameter count, MAC, and inference speed use this command:
```
python CollegiateDataClub/basicsr/models/archs/CGNetMultiHead_arch.py
```

### Visualizing the training logs
* You can use Tensorboard to track the training status:
```
tensorboard --logdir=/CollegiateDataClub/logs
```
