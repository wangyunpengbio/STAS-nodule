from glob import glob
from os.path import join, exists

import pandas as pd


def main():
    print("Start")
    root_path = '/home/u18111510027/stas/stas-data-annotated'
    # nii文件列表
    nii_file_list = glob(join(root_path, "*", "*", "*nii.gz"))
    # csv文件列表
    csv_path = join(root_path, "annotation-info.csv")
    annotation_DF = pd.read_csv(csv_path, sep="\t")
    csv_file_list = []
    for row in annotation_DF.iterrows():
        ct_path = join(root_path, row[1]["Hospital"], row[1]["STAS_stat"], row[1]["Sample_name"] + ".nii.gz")
        seg_path = join(root_path, row[1]["Hospital"], row[1]["STAS_stat"], row[1]["Sample_name"] + "_seg.nii.gz")
        csv_file_list.append(ct_path)
        csv_file_list.append(seg_path)
        if not exists(ct_path): print(ct_path)
        if not exists(seg_path): print(seg_path)
    print("-------以下为未在csv文件里面标注征象的-------")
    for item in sorted(nii_file_list):
        if item not in csv_file_list: print(item)
    print(annotation_DF)


if __name__ == "__main__":
    main()
