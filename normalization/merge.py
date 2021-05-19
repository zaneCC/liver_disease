# 合并成汇总表

import pandas as pd
import numpy as np
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/')
import constants
'''
    所有症状-分型合并
'''
PATH_TONGUE = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/入院记录-舌脉象-病历表.xlsx'
PATH_SYMP = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/病历症状表（规范化）.xlsx'
PATH_DIAGNOSIS = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/分型数值化.xlsx'

# 所有症状-分型合并
def mergeSymp_Diagnosis():
    sympDF = pd.read_excel(PATH_SYMP)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)

    resultDF = pd.merge(diagnosisDF, sympDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    # resultDF.drop(['ZHENGHOU2'],axis=1,inplace=True)
    # resultDF.drop(['Unnamed: 0'],axis=1,inplace=True)
    return resultDF

# 所有症状-分型-舌脉象合并
def mergeSymp_Diagnosis_Tongue():
    sympDF = pd.read_excel(PATH_SYMP)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)
    tongueDF = pd.read_excel(PATH_TONGUE)

    combineTongueSympDF = pd.merge(tongueDF, sympDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF = pd.merge(diagnosisDF, combineTongueSympDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    # resultDF.drop(['ZHENGHOU2'],axis=1,inplace=True)
    # resultDF.drop(['Unnamed: 0'],axis=1,inplace=True)
    return resultDF

# 主症-伴随症-分型合并
def mergeSympMainAcc_Diagnosis():
    sympDF = pd.read_csv(constants.SYMP_MAIN_ACC_PATH)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)

    resultDF = pd.merge(sympDF, diagnosisDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    return resultDF

# 主症-伴随症-舌脉象-分型合并
def mergeSympMainAcc_Tongue_Diagnosis():
    sympDF = pd.read_csv(constants.SYMP_MAIN_ACC_PATH)
    tongueDF = pd.read_excel(PATH_TONGUE)
    diagnosisDF = pd.read_excel(PATH_DIAGNOSIS)

    combineTongueSympDF = pd.merge(sympDF, tongueDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF = pd.merge(combineTongueSympDF, diagnosisDF, how='left', on=constants.INHOSPTIAL_ID)
    resultDF.fillna(value=0, inplace=True)

    return resultDF

if __name__ == '__main__':
    resultDF = mergeSymp_Diagnosis()
    resultDF.to_csv(constants.MERGE_CSV_PATH)