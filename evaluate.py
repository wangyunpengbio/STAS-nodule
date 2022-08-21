import logging
import os
from os.path import join
import pathlib
import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Optional, Sequence, Tuple, Type, Union
import yaml
from lib.model.factory import get_model, get_scheduler
from lib.dataset.factory import get_transform
import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch, PersistentDataset
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AddChanneld, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, EnsureTyped, EnsureType
from monai.transforms import CropForegroundd, SpatialCropD, ResizeWithPadOrCropd, RandCropByLabelClassesd, Spacingd, DataStatsd, ScaleIntensityRanged
from monai.utils import first, set_determinism
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
from monai.data import CSVSaver

def evaluate_epoch(model, device, data_loader, saver):
    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2, num_classes=2)])
    auc_metric = ROCAUCMetric()
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        for val_data in data_loader:
            val_data = val_data[0]
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            outputs = model(val_images)
            outputs_argmax = outputs.argmax(dim=1)
            # saver.save_batch(outputs_argmax, val_data["img_meta_dict"]) # 要加上finalize才会写入，运行时候注释掉了
            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, val_labels], dim=0)

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        y_onehot = [post_label(i) for i in decollate_batch(y)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_metric_value = auc_metric.aggregate()
        auc_metric.reset()
        del y_pred_act, y_onehot
    return acc_metric, auc_metric_value

def parse_args():
    parse = argparse.ArgumentParser(description='STAS network evaluate. Demo: \n'
'python evaluate.py --tag 0130-resnet34-pretrained --ymlpath experiment/resnet34.yaml --train_prefix tumor_0 --cache_prefix persistent_cache --pth_name best_test')
    parse.add_argument('--tag', type=str, default='', dest='tag',
                       help='distinct from other try')
    parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                       help='config uesd to modify the default setting')
    parse.add_argument('--train_prefix', type=str, default='tumor_0', dest='train_prefix',
                       help='choose hospital to be the trainset and the other will be the test set: '
                            'train_prefix: tumor_[0,1,2,3,4] | zhongshan_[0,1,2,3,4]')
    parse.add_argument('--cache_prefix', type=str, default='persistent_cache', dest='cache_prefix',
                       help='the prefix to mark the cached transformed dataset, '
                            'the final cache dir will be `cache_prefix` + `_` + `train_prefix`')
    parse.add_argument('--pth_name', type=str, default='best_metric_model_classification3d_dict', dest='pth_name',
                       help='choose the pth to load: '
                            'best_metric_model_classification3d_dict|best_test|best_val|latest')
    args = parse.parse_args()
    return args

def main():
    monai.config.print_config()
    set_determinism(seed=2022)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    args = parse_args()

    # merge config with yaml
    if args.ymlpath is not None:
        cfg_from_yaml(args.ymlpath)  # use the parameters in the yaml file (use the parameters in the yaml file to replace the raw parameters in the config.py)
    # merge config with argparse
    opt = copy.deepcopy(cfg)
    opt = merge_dict_and_yaml(args.__dict__, opt)  # squeeze the EasyDict and merge with args _easy_dict_squeeze
    print_easy_dict(opt)

    # add data_augmentation
    transform_dict = get_transform(opt.transform_name)
    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2, num_classes=2)])

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # resnet18_defined = monai.networks.nets.resnet18(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model = get_model(opt.model_name, opt.pretrained)
    model = model.to(device)



    split_set_name = opt.train_prefix + '-MakrAs-Train'

    root_dir = '/home/u18111510027/stas'
    save_dir = join(root_dir, 'save_models', opt.tag)
    model.load_state_dict(torch.load(join(save_dir, '{}.pth'.format(opt.pth_name))))
    model.eval()


    data_path = '/home/u18111510027/stas/stas-data-annotated'
    # csv文件列表
    csv_path = join(data_path, "annotation-info-split.xlsx")
    annotation_DF = pd.read_csv(csv_path, sep="\t")

    label_to_np_dict = {'negative': 0, 'positive': 1}
    train_files, val_files, test_files = [], [], []
    for row in annotation_DF.iterrows():
        img = join(data_path, row[1]["Hospital"], row[1]["STAS_stat"], row[1]["Sample_name"] + ".nii.gz")
        seg = join(data_path, row[1]["Hospital"], row[1]["STAS_stat"], row[1]["Sample_name"] + "_seg.nii.gz")
        label = label_to_np_dict[row[1]["STAS_stat"]]
        if row[1][split_set_name] == 'Train':
            train_files.append({"img": img, "seg": seg, "label": label})
        elif row[1][split_set_name] == 'Val':
            val_files.append({"img": img, "seg": seg, "label": label})
        elif row[1][split_set_name] == 'Test':
            test_files.append({"img": img, "seg": seg, "label": label})
        else:
            raise KeyError('Dataset split should be included in Train | Val | Test')

    # 直接随机到结节中心
    persistent_cache = pathlib.Path(root_dir, opt.cache_prefix + "_" + opt.train_prefix)
    persistent_cache.mkdir(parents=True, exist_ok=True)

    # create a validation data loader
    val_ds = monai.data.PersistentDataset(data=val_files, transform=transform_dict['val_transforms'], cache_dir=persistent_cache)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=8, pin_memory=False)

    test_ds = monai.data.PersistentDataset(data=test_files, transform=transform_dict['val_transforms'], cache_dir=persistent_cache)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=8, pin_memory=False)

    val_acc_metric, val_auc_metric = evaluate_epoch(model, device, val_loader, CSVSaver(output_dir=save_dir, filename='best_metric_validation.csv'))
    test_acc_metric, test_auc_metric = evaluate_epoch(model, device, test_loader, CSVSaver(output_dir=save_dir, filename='best_metric_test.csv'))

    with open("summary.xlsx", 'a') as f:
        # f.write("\tval accuracy\tval AUC\ttest accuracy\ttest AUC\n")
        f.write('{}\t{}\t{}\t{}\t{}\n'.format('{}/{}.pth'.format(opt.tag, opt.pth_name),round(val_acc_metric*100,2),round(val_auc_metric*100,2),round(test_acc_metric*100,2),round(test_auc_metric*100,2)))



if __name__ == "__main__":
    main()
