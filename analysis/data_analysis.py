# 数据分析
OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
import pandas as pd
import numpy as np
import sys
from typing import Counter, Iterable
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants
import utils.misc as misc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix

# 分型数值化
# PATH = constants.PATH_DIAGNOSIS_NORMALIZATION
# 汇总表-未做特征选择
PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
# PATH = constants.SELECTION_MERGE_CSV_PATH

# 汇总表-SMOTE过采样后
# PATH = constants.SMOTE_MERGE_CSV_PATH
# SMOTE Borderline1 采样
# PATH = constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH
# SMOTE_D 采样
# PATH = constants.SMOTE_D_MERGE_CSV_PATH
# SMOTE_BORDERLINE_D 采样
# PATH = constants.SMOTE_Borderline_D_CSV_PATH

class DataAnalysis():


    def __init__(self):
        self.df = pd.read_csv(PATH)
        # self.df = pd.read_excel(PATH)

        self.misc = misc.Misc()
        print(self.df['ZHENGHOU1'].value_counts())

    def plotSize(self):
        self.misc.plotSize(self.df['ZHENGHOU1'])
        self.misc.show()

    # 统计症状的频次
    def countSym(self):
        dicSym = {}
        for index, row in self.df.iteritems():
            if index == 'INHOSPTIAL_ID' or index == 'ZHENGHOU1':
                continue
            
            # count_1 = 0; count_2 = 0
            # for row in self.df.itertuples():
            #     _value = getattr(row, index)
            #     _class = getattr(row, 'ZHENGHOU1')

            #     if _class == constants.CLASS_1 and _value == 1:
            #         count_1 += 1
            #     elif _class == constants.CLASS_2 and _value == 1:
            #         count_2 += 1
                

            # # 统计症状频次
            # dicSym[index] = {constants.CLASS_1:count_1,constants.CLASS_2:count_2}

            dicSym[index] = Counter(self.df[index])[1]

        return dicSym

    def countSymByZHENGHOU1(self):
        for row in self.df.itertuples():
            _ZHENGHOU1 = getattr(row, 'ZHENGHOU1') 
            
    
    def countZhengHou(self):
        count = self.df['ZHENGHOU1'].groupby(self.df['ZHENGHOU2']).describe()
        # r = self.df.groupby(['ZHENGHOU1','ZHENGHOU2']).groups
        grouped = self.df.groupby(['ZHENGHOU2']).groups
        print(Counter(self.df['ZHENGHOU1']))

        # 阳黄证
        z1 = {'证型':[],'频次':[]}
        # 阴阳黄证，阴阳黄
        z2 = {'证型':[],'频次':[]}
        # 阴黄证
        z3 = {'证型':[],'频次':[]}

        for row in self.df.itertuples():
            _ZHENGHOU1 = getattr(row, 'ZHENGHOU1') 
            _ZHENGHOU2 = getattr(row, 'ZHENGHOU2') 
            if _ZHENGHOU1 == '阳黄证':
                if _ZHENGHOU2 in z1['证型']:
                    index = z1['证型'].index(_ZHENGHOU2)
                    z1['频次'][index] += 1
                else:
                    z1['证型'].append(_ZHENGHOU2)
                    z1['频次'].append(1)
            elif _ZHENGHOU1 == '阴阳黄证' or _ZHENGHOU1 == '阴阳黄':
                if _ZHENGHOU2 in z2['证型']:
                    index = z2['证型'].index(_ZHENGHOU2)
                    z2['频次'][index] += 1
                else:
                    z2['证型'].append(_ZHENGHOU2)
                    z2['频次'].append(1)
            elif _ZHENGHOU1 == '阴黄证':
                if _ZHENGHOU2 in z3['证型']:
                    index = z3['证型'].index(_ZHENGHOU2)
                    z3['频次'][index] += 1
                else:
                    z3['证型'].append(_ZHENGHOU2)
                    z3['频次'].append(1)

        print(z1)
        print(z2)
        print(z3)

    def EDA(self):
        # corrdata = self.df.corr()
        # sns.heatmap(corrdata,annot=True)

        sns.countplot(x='class',data=self.df,hue='sudden weight loss')
        plt.show()

if __name__ == '__main__':
    dataAna = DataAnalysis()
    # 统计症状频次
    dicSym = dataAna.countSym()
    print(dicSym)
    # sortDic = sorted(dicSym.items(), key=lambda e:e[1], reverse=True)
    # print(sortDic[:40])

    # 分型统计
    # dataAna.countZhengHou()

    # 分型饼图
    # dataAna.plotSize()

    # EDA
    # dataAna.EDA()