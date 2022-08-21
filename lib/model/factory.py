
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim import lr_scheduler
from os.path import join
import torch
from monai.networks.nets import EfficientNetBN, ResNet, resnet10, resnet18, resnet34, resnet50
from typing import Any, Optional, Sequence, Tuple, Type, Union
import logging

PATH_PRETRAINED_WEIGHTS = "/home/u18111510027/stas/MedicalNet_pytorch_files/pretrain/"

def create_pretrained_medical_resnet(
    pretrained_path: str,
    model_constructor: callable = resnet18,
    spatial_dims: int = 3,
    n_input_channels: int = 1,
    num_classes: int = 2,
    **kwargs_monai_resnet: Any
) -> Tuple[ResNet, Sequence[str]]:
    """This si specific constructor for MONAI ResNet module loading MedicalNEt weights.
    See:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """
    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
    logging.debug(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
    logging.debug(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
    logging.debug(f"unused pretrained: {len(unused)}")
    assert len(inside) > len(missing)
    assert len(inside) > len(unused)

    pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net.load_state_dict(pretrain['state_dict'], strict=False)
    return net, inside

def get_model(model_name, pretrained=True):
    if pretrained:
        logging.info(f"load the pretrained model")
    if model_name == 'ResNet10':
        if pretrained:
            path_pretrained_weights = join(PATH_PRETRAINED_WEIGHTS, "resnet_10_23dataset.pth")
            model, pretraineds_layers = create_pretrained_medical_resnet(path_pretrained_weights,
                                                                         model_constructor=resnet10)
        else:
            model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2)
        return model
    if model_name == 'ResNet18':
        if pretrained:
            path_pretrained_weights = join(PATH_PRETRAINED_WEIGHTS, "resnet_18_23dataset.pth")
            model, pretraineds_layers = create_pretrained_medical_resnet(path_pretrained_weights,
                                                                         model_constructor=resnet18)
        else:
            model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
        return model
    elif model_name == 'ResNet34':
        if pretrained:
            path_pretrained_weights = join(PATH_PRETRAINED_WEIGHTS, "resnet_34_23dataset.pth")
            model, pretraineds_layers = create_pretrained_medical_resnet(path_pretrained_weights,
                                                                         model_constructor=resnet34)
        else:
            model = resnet34(spatial_dims=3, n_input_channels=1, num_classes=2)
        return model
    elif model_name == 'ResNet50':
        if pretrained:
            path_pretrained_weights = join(PATH_PRETRAINED_WEIGHTS, "resnet_50_23dataset.pth")
            model, pretraineds_layers = create_pretrained_medical_resnet(path_pretrained_weights,
                                                                         model_constructor=resnet50)
        else:
            model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
        return model
    else:
        raise KeyError('Model class should select from ResNet10|ResNet18|ResNet34|ResNet50')


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'invariant':
        def lambda_rule(epoch):
            lr_l = 1.0
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
