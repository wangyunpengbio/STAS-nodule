# CT-based fusion model via deep learning and machine learning to predict spread through air spaces in stage Ⅰ lung adenocarcinoma

## Background

Spread through air spaces (STAS), a new invasion pattern, is an adverse prognostic indicator of lung adenocarcinoma. This study aims to develop computed tomography (CT)-based fusion model via deep learning and traditional machine learning for predicting STAS in stage Ⅰ lung adenocarcinoma. 
**In this repository, we provide complete scripts for dataset splitting, model training/evaluating, and deep model feature extraction.**

![Introduction.png](https://github.com/wangyunpengbio/STAS-nodule/raw/master/resources/demo.jpg)

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

