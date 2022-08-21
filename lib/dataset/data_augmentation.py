
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from monai.transforms import Activations, AddChanneld, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, EnsureTyped, EnsureType
from monai.transforms import CropForegroundd, SpatialCropD, ResizeWithPadOrCropd, RandCropByLabelClassesd, Spacingd, DataStatsd, ScaleIntensityRanged
from monai.transforms import RandGaussianNoised, RandShiftIntensityd, RandGibbsNoised, RandAdjustContrastd, RandGaussianSharpend, RandGaussianSmoothd, RandCoarseDropoutd

import torch
import numpy as np

def get_transform():
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AddChanneld(keys=["img", "seg"]),
            Spacingd(keys=["img", "seg"], pixdim=(1, 1, 1)),
            ScaleIntensityRanged(keys=["img"], a_min=-1200, a_max=300, b_min=0, b_max=1, clip=True),
            RandCropByLabelClassesd(keys=["img", "seg"], label_key='seg', spatial_size=[48, 48, 48], ratios=[0, 1],
                                    num_classes=2),
            ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=(48, 48, 48)),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 2)),
            EnsureTyped(keys=["img", "seg"]),
            # DataStatsd(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AddChanneld(keys=["img", "seg"]),
            Spacingd(keys=["img", "seg"], pixdim=(1, 1, 1)),
            ScaleIntensityRanged(keys=["img"], a_min=-1200, a_max=300, b_min=0, b_max=1, clip=True),
            RandCropByLabelClassesd(keys=["img", "seg"], label_key='seg', spatial_size=[48, 48, 48], ratios=[0, 1],
                                    num_classes=2),
            ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=(48, 48, 48)),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    # post_pred = Compose([EnsureType(), Activations(softmax=True)])
    # post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2, num_classes=2)])
    transform_dict = {
        'train_transforms': train_transforms,
        'val_transforms': val_transforms,
        # 'post_pred': post_pred,
        # 'post_label': post_label
    }
    return transform_dict


def get_intensity_transform():
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AddChanneld(keys=["img", "seg"]),
            Spacingd(keys=["img", "seg"], pixdim=(1, 1, 1)),
            ScaleIntensityRanged(keys=["img"], a_min=-1200, a_max=300, b_min=0, b_max=1, clip=True),
            RandShiftIntensityd(keys=["img"], offsets=(-50,-50), prob=0.2),
            RandCropByLabelClassesd(keys=["img", "seg"], label_key='seg', spatial_size=[48, 48, 48], ratios=[0, 1],
                                    num_classes=2),
            RandGaussianNoised(keys=["img"], prob=0.1),
            RandGibbsNoised(keys=["img"], prob=0.1),
            RandGaussianSmoothd(keys=["img"], prob=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.1),
            # RandGaussianSharpend(keys=["img"], prob=0.1),
            RandCoarseDropoutd(keys=["img"], fill_value=0, holes=3, spatial_size=8, prob=0.2),
            ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=(48, 48, 48)),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 2)),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AddChanneld(keys=["img", "seg"]),
            Spacingd(keys=["img", "seg"], pixdim=(1, 1, 1)),
            ScaleIntensityRanged(keys=["img"], a_min=-1200, a_max=300, b_min=0, b_max=1, clip=True),
            RandCropByLabelClassesd(keys=["img", "seg"], label_key='seg', spatial_size=[48, 48, 48], ratios=[0, 1],
                                    num_classes=2),
            ResizeWithPadOrCropd(keys=["img", "seg"], spatial_size=(48, 48, 48)),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    # post_pred = Compose([EnsureType(), Activations(softmax=True)])
    # post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2, num_classes=2)])
    transform_dict = {
        'train_transforms': train_transforms,
        'val_transforms': val_transforms,
        # 'post_pred': post_pred,
        # 'post_label': post_label
    }
    return transform_dict
# ##########################################
# Test
# ##########################################
def main():
    print(1)


if __name__ == '__main__':
    main()
