import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from xpinyin import Pinyin
from collections import defaultdict
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from glob import glob
from os.path import join
import itertools
from tqdm import tqdm
import argparse

pd.set_option('display.max_columns', None)

def calculate_auc(clf, x, y_true):
    clf.predict(x)

    y_score = clf.predict_proba(x)
    y_score = y_score[:, 1]

    roc_score = roc_auc_score(y_true, y_score)
    return roc_score


def calculate_roc(dataset, train_col, C, kernel, whether_using_dl_feature):
    roc_dict = defaultdict(list)
    addition_scaler_list = ["out_" + str(i) for i in range(1)] #+ ["in_"+str(i) for i in range(2048)]
    scaler_list = ['age', 'type', 'z_real', 'min_real', 'max_real']
    if whether_using_dl_feature:
        scaler_list = scaler_list + addition_scaler_list
    label_list = ['Hospital', 'sex', 'location', 'spiculated', 'lobulated', 'cavity', 'vacuole', 'boundary',
                  'airbronchogram', 'vpi', 'vessel', 'lymphadenovarix']

    sc = StandardScaler()
    sc_array = sc.fit_transform(dataset.loc[:, scaler_list])
    # ohe = OneHotEncoder(drop='first')
    ohe = OneHotEncoder()
    oh_array = ohe.fit_transform(dataset.loc[:, label_list]).toarray()
    # ohe_y = OneHotEncoder(drop='first')
    # oh_y_array = ohe.fit_transform(dataset.loc[:,'STAS_stat'])

    X = np.concatenate((sc_array, oh_array), axis=1)
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train = X[(dataset[train_col] == 'Train')]
    X_val = X[dataset[train_col] == 'Val']
    X_test = X[dataset[train_col] == 'Test']
    y_train = y[(dataset[train_col] == 'Train')]
    y_val = y[dataset[train_col] == 'Val']
    y_test = y[dataset[train_col] == 'Test']

    svm_cls = svm.SVC(C=C, kernel=kernel, probability=True, random_state=2022)
    svm_cls.fit(X_train, y_train)

    train_auc_rf = calculate_auc(svm_cls, X_train, y_train)
    val_auc_rf = calculate_auc(svm_cls, X_val, y_val)
    test_auc_rf = calculate_auc(svm_cls, X_test, y_test)
    roc_dict['svm'].extend([train_auc_rf, val_auc_rf, test_auc_rf])

    roc_dict['split'].extend(['train', 'val', 'test'])

    return roc_dict

# 目前是直接把文件夹里面的Deep learning跑出来的特征，直接按照样本编号append到一起去，忽略了Deep learning时候的数据集划分
# 这种做法的缺陷是，如果dl的划分方式跟randomforest不一样，会有数据泄露的可能性
# 但是目前我都用的同一个数据划分表csv，所以不用担心这个问题
def read_DL_data(target_tagdir_list):
    dl_df = pd.read_csv(target_tagdir_list[0])
    for item in target_tagdir_list[1:]:
        dl_df = dl_df.append(pd.read_csv(item))
    return dl_df

def parse_args():
    parse = argparse.ArgumentParser(description='Support Vector Machine + Deep Learning model')
    parse.add_argument('--tag_id', type=int, default=11, dest='tag_id',
                       help='define the target DeepLearning experiment ID')
    parse.add_argument('--dl_feature', dest='dl_feature', action='store_true')
    parse.add_argument('--no_dl_feature', dest='dl_feature', action='store_false')
    parse.set_defaults(dl_feature=True)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    p = Pinyin()
    # Importing the dataset
    root_dir = '/home/u18111510027/stas/'
    out_dir = join(root_dir, 'rf-result-tta')
    out_dir = join(root_dir, 'rf-result')
    dataset = pd.read_csv(join(root_dir, 'stas-data-annotated/annotation-info-split.xlsx'), sep='\t')
    dataset.age = [int(item[1:3]) for item in dataset.age]
    dataset.location = [p.get_pinyin(item) for item in dataset.location]

    whether_using_dl_feature = args.dl_feature

    tag_id = args.tag_id
    assert tag_id % 2 == 1, "Error, MUST be the odd number"
    result_dict = defaultdict(list)
    target_epoch_list = ['best_val', 'best_test', 'latest']
    C_list = [1, 2, 3, 4, 5]
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']

    param_grid_list = list(itertools.product(C_list, kernel_list))
    for (C, kernel) in tqdm(param_grid_list):
        for tag in [tag_id, tag_id + 1]:
            hospital = 'tumor' if tag % 2 == 1 else 'zhongshan'
            for i in range(5): # 指定fold数目
                for target_epoch in target_epoch_list:
                    target_tagdir_list = glob(join(root_dir, 'save_models', '0{}{}*'.format(tag, i), 'tta_{}_*.csv'.format(target_epoch)))
                    target_tagdir_list = glob(join(root_dir, 'save_models', '0{}{}*'.format(tag, i), '{}_*.csv'.format(target_epoch)))
                    assert len(target_tagdir_list) == 3, "When finish get_intermediate_layer.py, there should be train, val and test CSV."
                    dl_df = read_DL_data(target_tagdir_list)
                    dataset['tradition_id'] = dataset['Hospital'] + '/' + dataset['STAS_stat'] + '/' + dataset['Sample_name']
                    # 然后train的时候把列给选中！
                    dataset_merged = dataset.merge(dl_df, left_on='tradition_id', right_on='Unnamed: 0')
                    train_col = '{}_{}-MakrAs-Train'.format(hospital, i)
                    roc_dict = calculate_roc(dataset_merged, train_col, C, kernel, whether_using_dl_feature)
                    for key, value_list in roc_dict.items():
                        result_dict[key].extend(value_list)
                    for item in range(len(value_list)):
                        result_dict['fold'].append(i)
                        result_dict['hospital'].append(hospital)
                        result_dict['epoch'].append(target_epoch)
                        result_dict['C'].append(C)
                        result_dict['kernel'].append(kernel)
    result_df = pd.DataFrame.from_dict(result_dict)
    if whether_using_dl_feature:
        tag_id = "{}_dl".format(tag_id)
    else:
        tag_id = "{}_nodl".format(tag_id)
    result_df.to_csv(join(out_dir, "all_svm_0{}.xlsx".format(tag_id)), sep='\t')
    summarized_df = result_df.groupby(["hospital", "epoch", "split", "C", "kernel"])[['svm']].mean()
    # summarized_df = summarized_df.reset_index(level=[0, 1, 2])
    summarized_df.to_csv(join(out_dir, "mean_svm_0{}.xlsx".format(tag_id)), sep='\t')
    print(summarized_df)
