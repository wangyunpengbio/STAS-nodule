
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_transform(transform_name):
    if transform_name == 'baseline':  # 我自己新加的
        # import the data augmentation class
        from .data_augmentation import get_transform
        transform_dict = get_transform()
        return transform_dict
    elif transform_name == 'intensity':  # 我自己新加的
        # import the data augmentation class
        from .data_augmentation import get_intensity_transform
        transform_dict = get_intensity_transform()
        return transform_dict
    else:
        raise KeyError('Dataset class should select from baseline / ')
