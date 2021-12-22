
OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'

import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# # define example
# # data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# data = ['warm', 'cold', 'hot', 'hot', 'warm', 'warm', 'hot']

# values = array(data)
# print(values)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# # invert first example
# # inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

# r = [argmax(onehot_encoded[0, :])]
# print(r)

PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/output/入院记录-舌脉象-病历表.xlsx'
df = pd.read_excel(PATH)

df.drop(['INHOSPTIAL_ID'],axis=1,inplace=True)

data1 = pd.get_dummies(df["舌体"])
# res  = df.join(data1)
print(data1)
