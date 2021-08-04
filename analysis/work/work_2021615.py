OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
from itertools import count
import pandas as pd
import numpy as np
import sys
from typing import Counter, Iterable
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
import constants
import utils.misc as misc
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/analysis/imbalance/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import SMOTE, SMOTE_Borderline1, SMOTE_D, SMOTE_Borderline_D
from sklearn.datasets import make_classification

# 汇总表-未做特征选择
PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
# PATH = constants.SELECTION_MERGE_CSV_PATH
# SMOTE 采样
# TO_PATH = constants.SMOTE_MERGE_CSV_PATH
# SMOTE Borderline1 采样
# TO_PATH = constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH
# SMOTE_D 采样
# TO_PATH = constants.SMOTE_D_MERGE_CSV_PATH
# SMOTE_BORDERLINE_D 采样
TO_PATH = constants.SMOTE_Borderline_D_CSV_PATH

def read_data():
    df = pd.read_csv(PATH)
    

    del df['INHOSPTIAL_ID']
    return df

if __name__ == '__main__':
    df = read_data()
    data = df.to_numpy()

    balanced_data_arr2 = ''
    if TO_PATH == constants.SMOTE_MERGE_CSV_PATH: # SMOTE 采样
        balanced_data_arr2 = SMOTE.SMOTE(data)
    elif TO_PATH == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH: # SMOTE Borderline1 采样
        balanced_data_arr2 = SMOTE_Borderline1.Border_SMOTE(data)
    elif TO_PATH == constants.SMOTE_D_MERGE_CSV_PATH:  # SMOTE_D 采样
        balanced_data_arr2 = SMOTE_D.SMOTE_D(data)
    elif TO_PATH == constants.SMOTE_Borderline_D_CSV_PATH:  # SMOTE_Borderline_D 采样
        balanced_data_arr2 = SMOTE_Borderline_D.SMOTE_Borderline_D(data)

    cols = df.columns.values.tolist()
    to_df = pd.DataFrame(data=balanced_data_arr2, columns=cols)
    to_df[to_df.iloc[:,:-1]>=0.5] = 1
    to_df[to_df.iloc[:,:-1]<0.5] = 0
    # print(to_df)
    # # # 导出文件
    to_df.to_csv(TO_PATH, index=False)
    