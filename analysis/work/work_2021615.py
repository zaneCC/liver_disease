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
from imblearn.over_sampling import RandomOverSampler

# 汇总表-未做特征选择
PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
# PATH = constants.SELECTION_MERGE_CSV_PATH
# 随机过采样
TO_PATH = constants.RANDOM_OVER_SAMPLER_CSV_PATH
# SMOTE 采样
# TO_PATH = constants.SMOTE_MERGE_CSV_PATH
# SMOTE Borderline1 采样
# TO_PATH = constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH
# SMOTE_D 采样
# TO_PATH = constants.SMOTE_D_MERGE_CSV_PATH
# SMOTE_BORDERLINE_D 采样
# TO_PATH = constants.SMOTE_Borderline_D_CSV_PATH

def read_data():
    df = pd.read_csv(PATH)
    

    del df['INHOSPTIAL_ID']
    return df

# 进行 k 次采样，形成 k 次数据集
def resampling(k=1):

    df = read_data()
    data = df.to_numpy()

    if k == 1:
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
        cols.remove('ZHENGHOU1')
        cols.append('tag')
        cols.append('ZHENGHOU1')

        to_df = pd.DataFrame(data=balanced_data_arr2, columns=cols)
        to_df[to_df.iloc[:,:-1]>=0.5] = 1
        to_df[to_df.iloc[:,:-1]<0.5] = 0
        # print(to_df)
        # # # 导出文件
        to_df.to_csv(TO_PATH, index=False)
        return

    ''' 
        SMOTE:              SMOTE_MERGE_COMPARE_CSV_PATH
        SMOTE Borderline1:  SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH
        SMOTE_D:            SMOTE_D_MERGE_COMPARE_CSV_PATH
        SMOTE_BORDERLINE_D: SMOTE_Borderline_D_COMPARE_CSV_PATH
    '''
    for j in range(k):
        RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/随机过采样-采样-汇总表.csv'
        # SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE-采样-汇总表.csv'
        # SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_Borderline1-采样-汇总表.csv'
        # SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_D-采样-汇总表.csv'
        # SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_Borderline_D-采样-汇总表.csv'
                
        ''' 修改论文数据的时候使用 '''
        # RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ constants.PAPER_VERSION +'/对比实验/'+ str(j) +'/随机过采样-采样-汇总表.csv'
        # SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ constants.PAPER_VERSION +'/对比实验/'+ str(j) +'/SMOTE-采样-汇总表.csv'
        # SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ constants.PAPER_VERSION +'/对比实验/'+ str(j) +'/SMOTE_Borderline1-采样-汇总表.csv'
        # SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ constants.PAPER_VERSION +'/对比实验/'+ str(j) +'/SMOTE_D-采样-汇总表.csv'
        # SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ constants.PAPER_VERSION +'/对比实验/'+ str(j) +'/SMOTE_Borderline_D-采样-汇总表.csv'
        
        path = constants.RANDOM_OVER_SAMPLER_CSV_PATH
        to_path = RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH

        balanced_data_arr2 = ''
        if path == constants.SMOTE_MERGE_CSV_PATH: # SMOTE 采样
            balanced_data_arr2 = SMOTE.SMOTE(data)
        elif path == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH: # SMOTE Borderline1 采样
            balanced_data_arr2 = SMOTE_Borderline1.Border_SMOTE(data)
        elif path == constants.SMOTE_D_MERGE_CSV_PATH:  # SMOTE_D 采样
            balanced_data_arr2 = SMOTE_D.SMOTE_D(data)
        elif path == constants.SMOTE_Borderline_D_CSV_PATH:  # SMOTE_Borderline_D 采样
            balanced_data_arr2 = SMOTE_Borderline_D.SMOTE_Borderline_D(data)
        elif path == constants.RANDOM_OVER_SAMPLER_CSV_PATH:  # 随机过采样
            cols = df.columns.values.tolist()
            X = data[:,:-1]
            y = data[:,-1]
            X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
            balanced_data_arr2 = np.column_stack((X_resampled, y_resampled))
            to_df = pd.DataFrame(data=balanced_data_arr2, columns=cols)
            to_df.to_csv(to_path, index=False)
            continue

        cols = df.columns.values.tolist()
        # cols.remove('ZHENGHOU1')
        # cols.append('tag')
        cols.append('ZHENGHOU1')

        to_df = pd.DataFrame(data=balanced_data_arr2, columns=cols)
        to_df[to_df.iloc[:,:-1]>=0.5] = 1
        to_df[to_df.iloc[:,:-1]<0.5] = 0
        # print(to_df)
        # # # 导出文件
        to_df.to_csv(to_path, index=False)



if __name__ == '__main__':
    
    resampling(k=6)

    # resampling()

    # df = read_data()
    # data = df.to_numpy()

    # balanced_data_arr2 = ''
    # if TO_PATH == constants.SMOTE_MERGE_CSV_PATH: # SMOTE 采样
    #     balanced_data_arr2 = SMOTE.SMOTE(data)
    # elif TO_PATH == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH: # SMOTE Borderline1 采样
    #     balanced_data_arr2 = SMOTE_Borderline1.Border_SMOTE(data)
    # elif TO_PATH == constants.SMOTE_D_MERGE_CSV_PATH:  # SMOTE_D 采样
    #     balanced_data_arr2 = SMOTE_D.SMOTE_D(data)
    # elif TO_PATH == constants.SMOTE_Borderline_D_CSV_PATH:  # SMOTE_Borderline_D 采样
    #     balanced_data_arr2 = SMOTE_Borderline_D.SMOTE_Borderline_D(data)

    # cols = df.columns.values.tolist()
    # to_df = pd.DataFrame(data=balanced_data_arr2, columns=cols)
    # to_df[to_df.iloc[:,:-1]>=0.5] = 1
    # to_df[to_df.iloc[:,:-1]<0.5] = 0
    # # print(to_df)
    # # # # 导出文件
    # to_df.to_csv(TO_PATH, index=False)
    