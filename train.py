import argparse
import copy
import logging
import os
import pathlib
import shutil
import sys
from os.path import join

import monai
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_transform
from lib.model.factory import get_model, get_scheduler


def evaluate_epoch(model, device, data_loader):
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
            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
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
    parse = argparse.ArgumentParser(description='STAS network')
    parse.add_argument('--tag', type=str, default='', dest='tag',
                       help='distinct from other try')
    parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                       help='config uesd to modify the default setting')
    parse.add_argument('--train_prefix', type=str, default='tumor_0', dest='train_prefix',
                       help='choose hospital to be the trainset and the other will be the test set: '
                            'train_prefix: tumor_[0,1,2,3,4] | zhongshan_[0,1,2,3,4]')
    parse.add_argument('--cache_prefix', type=str, default='persistent_cache', dest='cache_prefix',
                       help='the prefix to mark the cached transformed dataset, '
                            'the final cache dir will be cache_prefix + train_prefix')
    args = parse.parse_args()
    return args


def main():
    monai.config.print_config()
    set_determinism(seed=2022)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    args = parse_args()

    # merge config with yaml
    if args.ymlpath is not None:
        cfg_from_yaml(
            args.ymlpath)  # use the parameters in the yaml file (use the parameters in the yaml file to replace the raw parameters in the config.py)
    # merge config with argparse
    opt = copy.deepcopy(cfg)
    opt = merge_dict_and_yaml(args.__dict__, opt)  # squeeze the EasyDict and merge with args _easy_dict_squeeze
    print_easy_dict(opt)

    # add data_augmentation
    transform_dict = get_transform(opt.transform_name)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # resnet18_defined = monai.networks.nets.resnet18(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model = get_model(opt.model_name, opt.pretrained)
    model = model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = get_scheduler(optimizer, opt)

    split_set_name = opt.train_prefix + '-MakrAs-Train'

    root_dir = '/home/u18111510027/stas'
    save_dir = join(root_dir, 'save_models', args.tag)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(args.ymlpath, join(save_dir, "config.yaml"))

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

    # create a training data loader
    train_ds = monai.data.PersistentDataset(data=train_files, transform=transform_dict['train_transforms'],
                                            cache_dir=persistent_cache)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=8,
                              pin_memory=False)

    # create a validation data loader
    val_ds = monai.data.PersistentDataset(data=val_files, transform=transform_dict['val_transforms'],
                                          cache_dir=persistent_cache)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=8, pin_memory=False)

    test_ds = monai.data.PersistentDataset(data=test_files, transform=transform_dict['val_transforms'],
                                           cache_dir=persistent_cache)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=8, pin_memory=False)

    best_val_metric, best_test_metric = -1, -1
    best_val_metric_epoch, best_test_metric_epoch = -1, -1
    epoch_num = opt.niter + opt.niter_decay
    writer = SummaryWriter(log_dir=save_dir)
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        writer.add_scalar("learning rate", lr, epoch + 1)
        for batch_data in train_loader:
            batch_data = batch_data[0]
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        train_acc_metric, train_auc_metric = evaluate_epoch(model, device, train_loader)
        val_acc_metric, val_auc_metric = evaluate_epoch(model, device, val_loader)
        test_acc_metric, test_auc_metric = evaluate_epoch(model, device, test_loader)

        writer.add_scalars("Accuracy",
                           {"Train": train_acc_metric, "Validation": val_acc_metric, "Test": test_acc_metric},
                           epoch + 1)
        writer.add_scalars("AUC",
                           {"Train": train_auc_metric, "Validation": val_auc_metric, "Test": test_auc_metric},
                           epoch + 1)
        if test_auc_metric > best_test_metric:
            best_test_metric = test_auc_metric
            best_test_metric_epoch = epoch + 1
            torch.save(model.state_dict(), join(save_dir, 'best_test.pth'))

        if val_auc_metric > best_val_metric:
            best_val_metric = val_auc_metric
            best_val_metric_epoch = epoch + 1
            torch.save(model.state_dict(), join(save_dir, 'best_val.pth'))
            print("saved new best metric model")
        print(
            "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, val_acc_metric, val_auc_metric, best_val_metric, best_val_metric_epoch
            )
        )

        scheduler.step()
        torch.save(model.state_dict(), join(save_dir, 'latest.pth'))
    print(f"train completed, best_val_metric: {best_val_metric:.4f} at epoch: {best_val_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
