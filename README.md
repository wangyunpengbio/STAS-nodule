# CT-based fusion model via deep learning and machine learning to predict spread through air spaces in stage Ⅰ lung adenocarcinoma

## Background

Spread through air spaces (STAS), a new invasion pattern, is an adverse prognostic indicator of lung adenocarcinoma. This study aims to develop computed tomography (CT)-based fusion model via deep learning and traditional machine learning for predicting STAS in stage Ⅰ lung adenocarcinoma. 
**In this repository, we provide complete scripts for dataset splitting, model training/evaluating, and deep model feature extraction.**

![Introduction.png](https://github.com/wangyunpengbio/STAS-nodule/raw/master/resources/demo.png)

## Requirement
Related packages:
```
Python >= 3.7
torch == 1.9.0
cuda >= 11.0
monai >= 0.8.1
pydicom
```
The complete list of packages can be found in `requirements.txt`. These packages are relatively stable, so you do not need to have your python environment to be consistent with `requirement.txt`. However, we still strongly recommend using anaconda for environment configuration.
```
conda create -n stas python=3.7
conda activate stas
pip install -r requirements.txt
```
Please remember to add the project directory to the `$PYTHONPATH` environment variable before using the repository, otherwise you will get the error `ModuleNotFoundError: No module named 'lib'`. For example, on a Linux system, the following command is required.

`export PYTHONPATH="/home/username/STAS-nodule:$PYTHONPATH"`

## Code Structure
- dl-features: Scripts used to get the features of the deep learning model.
- experiment: Directory of model configurations.
- lib: Scripts related to dataset splitting, data pre-processing, deep learning model training schedule, etc.
- ml-methods: Scripts related to machine learning models and fusion models. 
- stas-data-annotated: We provide four samples of data for demonstration and the files can be downloaded from Google Cloud Drive.
  - Google drive：
    `https://drive.google.com/file/d/1kriohALiY_3jx5ju9fFBfBTQYTnORmw3/view?usp=sharing`

## Usage

### Prerequisite
Before using the code in this repository, you should install the required packages, and then download the dataset from the cloud storage or use your own dataset. The dataset should be put into the directory `stas-data-annotated` as mentioned above in the 'Code Structure' section if you follow this tutorial.

### Step 1: Model Training
After the runtime environment and dataset are prepared, the training can be carried out via the `train.py` file. You need to specify the tag, model configuration, dataset splitting plan and the cache folder for that experiment.
```
(py37) [xxxxx@gpu1 STAS-nodule]$ python train.py -h
usage: train.py [-h] [--tag TAG] [--ymlpath YMLPATH]
                [--train_prefix TRAIN_PREFIX] [--cache_prefix CACHE_PREFIX]

STAS network

optional arguments:
  -h, --help            show this help message and exit
  --tag TAG             distinct from other try
  --ymlpath YMLPATH     config uesd to modify the default setting
  --train_prefix TRAIN_PREFIX
                        choose hospital to be the trainset and the other will
                        be the test set: train_prefix: tumor_[0,1,2,3,4] |
                        zhongshan_[0,1,2,3,4]
  --cache_prefix CACHE_PREFIX
                        the prefix to mark the cached transformed dataset, the
                        final cache dir will be cache_prefix + train_prefix
```
### Step 2: Model Evaluation
After training, model evaluation can be carried out using the `evaluate.py` file. This can be done by simply specifying one more piece of information (i.e. the weight of the model).
```
(py37) [xxxxx@gpu1 STAS-nodule]$ python evaluate.py -h
usage: evaluate.py [-h] [--tag TAG] [--ymlpath YMLPATH]
                   [--train_prefix TRAIN_PREFIX] [--cache_prefix CACHE_PREFIX]
                   [--pth_name PTH_NAME]

STAS network evaluate. Demo: python evaluate.py --tag 0130-resnet34-pretrained
--ymlpath experiment/resnet34.yaml --train_prefix tumor_0 --cache_prefix
persistent_cache --pth_name best_test

optional arguments:
  -h, --help            show this help message and exit
  --tag TAG             distinct from other try
  --ymlpath YMLPATH     config uesd to modify the default setting
  --train_prefix TRAIN_PREFIX
                        choose hospital to be the trainset and the other will
                        be the test set: train_prefix: tumor_[0,1,2,3,4] |
                        zhongshan_[0,1,2,3,4]
  --cache_prefix CACHE_PREFIX
                        the prefix to mark the cached transformed dataset, the
                        final cache dir will be `cache_prefix` + `_` +
                        `train_prefix`
  --pth_name PTH_NAME   choose the pth to load: best_metric_model_classificati
                        on3d_dict|best_test|best_val|latest
```

**If you find this repository useful, you can give it a `STAR`. Thank you.**