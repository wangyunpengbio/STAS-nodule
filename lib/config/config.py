
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from easydict import EasyDict
import os
import numpy as np

__C = EasyDict()
cfg = __C

# Data
__C.DATASET = EasyDict()
# the dir to save the cached transformed dataset
# __C.DATASET.PERSISTENT_CACHE = 'persistent_cache'
# choose the transform from the dataset/factory.py
__C.DATASET.transform_name = 'baseline'
# choose the train set and the other will be the test set: tumor|zhongshan
# __C.DATASET.train_set = 'tumor'

# TRAIN
__C.TRAIN = EasyDict()
# learning rate policy: lambda|step|plateau
__C.TRAIN.lr_policy = 'lambda'
# num of epoch that learning rate stay at initial
__C.TRAIN.niter = 15
# num of epoch that learning rate decay
__C.TRAIN.niter_decay = 25
# batch size
__C.TRAIN.batch_size = 16
# initial learning rate for adam
__C.TRAIN.lr = 0.0001
# momentum term of adam
__C.TRAIN.beta1 = 0.5
__C.TRAIN.beta2 = 0.9

# MODEL
__C.MODEL = EasyDict()
__C.MODEL.model_name = 'ResNet50'
__C.MODEL.pretrained = True

def cfg_from_yaml(filename):
    '''
    Load a config file and merge it into the default options
    Use the parameters in the yaml file to replace the raw parameters in the config.py
    :param filename:
    :return:
    '''
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = EasyDict(yaml.safe_load(f))
    _merge_a_into_b(yaml_cfg, __C)


def print_easy_dict(easy_dict):
    print('===' * 10)
    print('====YAML Parameters')
    for k, v in easy_dict.__dict__.items():
        print('{}: {}'.format(k, v))
    print('===' * 10)


def merge_dict_and_yaml(in_dict, easy_dict):
    if type(easy_dict) is not EasyDict:
        return in_dict
    easy_list = _easy_dict_squeeze(easy_dict)
    for (k, v) in easy_list:
        if k in in_dict:
            raise KeyError('The same Key appear {}/{}'.format(k, k))
    out_dict = EasyDict(dict(easy_list + list(in_dict.items())))
    return out_dict


def _easy_dict_squeeze(easy_dict):
    if type(easy_dict) is not EasyDict:
        print('Not EasyDict!!!')
        return []

    total_list = []
    for k, v in easy_dict.items():
        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                total_list += _easy_dict_squeeze(v)
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            total_list.append((k, v))
    return total_list


def _merge_a_into_b(a, b):
    '''
    Merge easyDict a to easyDict b
    :param a: from easyDict
    :param b: to easyDict
    :return:
    '''
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # check k in a or not
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].type)
            else:
                raise ValueError('Type mismatch ({} vs. {})'
                                 'for config key: {}'.format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
