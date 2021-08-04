# 入院-舌脉象
import pandas as pd
import numpy as np
import re
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/')
import constants
import utils.re_utils as re_utils

PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/data/入院记录——舌脉.csv'
TO_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/output/入院记录-舌脉象-病历表.xlsx'


INCASE_DICT = ['中医四诊舌体情况', '中医四诊舌下脉络情况','中医四诊舌质情况','中医四诊舌苔情况','中医四诊脉象情况']
TONGUE_VEINS = ['舌体','舌下脉络','舌质','舌苔','脉象']
# 分类前要清洗数据（把所有逗号/句号去掉），不在分类中的用0表示
# 舌体种类
KIND_TONGUE_POSTURE = {1:'正常|正常正常', 2:'瘦', 3:'胖大', 4:'边有齿痕|正常边有齿痕', 5:'胖大边有齿痕'}
# 舌质种类
KIND_TONGUE_COLOR = {1:'红', 2:'淡红', 3:'暗红|红暗', 4:'淡白', 5:'暗紫',
                     6:'红暗紫', 7:'淡暗红', 8:'暗紫红', 9:'暗' , 10:'红嫩',
                     11:'淡暗|暗淡', 12:'淡', 13:'红绛', 14:'淡红嫩', 15:'稍红'} 
# 脉象种类
KIND_TONGUE_PULSE = {1:'弦|弦弦', 2:'弦滑|滑弦', 3:'弦数', 4:'弦细', 5:'细弱', 6:'沉细', 7:'实弦', 8:'细数弦', 9:'实', 
                    10:'细', 11:'弦弱', 12:'细沉数弦', 13:'濡', 14:'弦缓', 15:'滑'} 
# 舌苔种类
KIND_TONGUE_FUR = {1:'白腻|腻白', 2:'黄腻|腻黄', 3:'黄', 4:'湿润白厚腻', 5:'白腻少', 6:'白腻厚|白厚腻',
                    7:'黄少', 8:'黄腻少', 9:'腻厚黄|黄腻厚|厚腻黄|黄厚腻.', 10:'黄厚', 11:'少腻', 12:'湿润黄少', 13:'腻',
                    14:'白湿润腻', 15:'白', 16:'厚腻湿润', 17:'黄干燥少', 18:'腐黄腻', 19:'厚腻腐黄白', 20:'湿润黄腻',
                    21:'腻湿润', 22:'湿润', 23:'白黄腻|白腻黄', 24:'白微腻', 25:'薄白', 26:'少', 27:'少白',
                    28:'干燥灰黑腻黄', 29:'薄黄腻|腻薄黄', 30:'白灰黑腻', 31:'薄黄厚腻', 32:'腻略黄', 33:'白腻中间有裂痕',
                    34:'白厚', 35:'少薄黄', 36:'黄干燥'} 
# 舌下脉络种类
KIND_SUBLINGUAL_VEIN = {1:'脉络正常|舌下脉络正常', 2:'脉络迂曲', 3:'舌下脉络增粗延长|脉络增粗延长', 4:'瘀斑或瘀条',
                    5:'瘀点', 6:'脉络增粗', 7:'瘀点脉络迂曲'} 

NOT_TONGUE_VEINS_DICT = {'{.'}

df = pd.read_csv(PATH)

# 初始化入院舌脉象表
ids = list(set(df[constants.INHOSPTIAL_ID]))
data = np.zeros((len(ids),len(TONGUE_VEINS)), dtype=int)
resultDF = pd.DataFrame(data=data, columns=TONGUE_VEINS, index=ids)

def getValue(_incaseName):
    for i,value in enumerate(TONGUE_VEINS):
        if TONGUE_VEINS[i] in _incaseName:
            return TONGUE_VEINS[i]

# 遍历表，并记录
for row in df.itertuples():
    _ids = getattr(row, constants.INHOSPTIAL_ID)
    _incaseName = getattr(row, constants.INCASE_FIELD_NAME)
    _incaseValue = getattr(row, constants.INCASE_FIELD_VALUE)

    # 清洗数据（把所有逗号/句号去掉）
    if '。' in _incaseValue:
        _incaseValue = str(_incaseValue).replace('。','')
    if '，' in _incaseValue:
        _incaseValue = str(_incaseValue).replace('，','')
    
    # 清洗数据（把无用数据设置0）
    _incaseValue = re_utils.rematchFilter(_incaseValue,NOT_TONGUE_VEINS_DICT)

    _value = getValue(_incaseName)
    
   
    if _value == TONGUE_VEINS[0]: # 舌体数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_POSTURE)
    if _value == TONGUE_VEINS[1]:  # 舌下脉络数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_SUBLINGUAL_VEIN)
    if _value == TONGUE_VEINS[2]:  # 舌质数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_COLOR)
    if _value == TONGUE_VEINS[3]:  # 舌苔数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_FUR)
    if _value == TONGUE_VEINS[4]:  # 脉象数据数值化
        _incaseValue = re_utils.rematch(str(_incaseValue), KIND_TONGUE_PULSE)

    resultDF[_value][_ids] = _incaseValue # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
    

# 保存文件
resultDF.to_excel(TO_PATH)

    

