from glob import glob
from os.path import join, isdir
import random
import pandas as pd
import numpy as np

# 该程序的目的就是1、除去30mm以上的结节，2、拆分数据集
if __name__ == '__main__':
    np.random.seed(2022)
    num_split_fold = 5
    data_path = '/home/u18111510027/stas/stas-data-annotated'
    # csv文件列表
    csv_path = join(data_path, "annotation-info-merged-realLEN.xlsx")
    annotation_DF = pd.read_csv(csv_path, sep="\t")
    # 去除30mm以上的结节
    annotation_DF = annotation_DF.loc[annotation_DF.max_real <= 30, :]
    def mark_train_val_test(annotation_DF, hospital_name, num_split_fold):
        val_frac = 1 / num_split_fold
        indices = annotation_DF.loc[annotation_DF.Hospital == hospital_name].index
        indices = indices.to_list()
        np.random.shuffle(indices)
        # 拆分数据集
        val_split = int(val_frac * len(indices))
        for i in range(num_split_fold):
            mark_col = hospital_name.split('-')[0] + '_{}-MakrAs-Train'.format(i)
            annotation_DF.loc[:, mark_col] = 'Test'
            val_indices = indices[val_split * i: val_split * (i+1)]
            train_indices = indices[:val_split * i] + indices[val_split * (i+1):]
            # 标注数据集划分的信息
            annotation_DF.loc[train_indices, mark_col] = 'Train'
            annotation_DF.loc[val_indices, mark_col] = 'Val'

    mark_train_val_test(annotation_DF, 'tumor-hospital-STAS', num_split_fold)
    mark_train_val_test(annotation_DF, 'zhongshan-hospital-STAS', num_split_fold)
    print(annotation_DF)
    annotation_DF.to_csv(join(data_path, "annotation-info-split!!!.xlsx"), sep='\t', index=False)

